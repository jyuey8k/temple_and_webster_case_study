import pandas as pd
import numpy as np

from sklearn.model_selection   import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder, StandardScaler
from sklearn.metrics           import roc_auc_score, accuracy_score, mean_squared_error
from lightgbm                  import LGBMClassifier, LGBMRegressor
from scipy.stats               import randint, uniform


class DemandForecaster:
    """
    A flexible demand forecasting tool emulating thesklearn interface.

    Built this class to let you build either a two-stage intermittent demand model
    (classifier + regressor) or a single-stage continuous regressor.  

   
    """
    def __init__(self,
        products_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        group_key: str = 'product_id',
        date_col: str = 'timestamp',
        freq: str = 'W',
        intermittent: bool = True
    ):
        self.products_df     = products_df
        self.transactions_df = transactions_df
        self.inventory_df    = inventory_df
        self.group_key       = group_key
        self.date_col        = date_col
        self.freq            = freq
        self.intermittent    = intermittent  # use two‐stage model when True

        self.df_panel   = None
        self.df_train   = None
        self.df_test    = None

        self.X_tr_cls = self.X_te_cls = None
        self.y_tr_cls = self.y_te_cls = None
        self.X_tr_reg = self.X_te_reg = None
        self.y_tr_reg = self.y_te_reg = None

        self.clf_search = None
        self.reg_search = None
        self.results    = None

    def _to_period(self, df: pd.DataFrame, col: str) -> pd.Series:
        return (
            pd.to_datetime(df[col])
              .dt.to_period(self.freq)
              .apply(lambda r: r.start_time)
        )

    def _merge_group_key(self):
        if self.group_key == 'product_id':
            self.trans = self.transactions_df.copy()
            self.inv   = self.inventory_df.copy()
        else:
            mapping = (
                self.products_df
                    [[ 'product_id', self.group_key ]]
                    .drop_duplicates()
            )
            self.trans = self.transactions_df.merge(mapping,
                                                     on='product_id',
                                                     how='left')
            self.inv   = self.inventory_df.merge(mapping,
                                                 on='product_id',
                                                 how='left')

    def _build_panel_index(self) -> pd.DataFrame:
        keys    = self.products_df[self.group_key].unique()
        periods = self._to_period(self.trans, self.date_col).unique()
        return (
            pd.MultiIndex
              .from_product([keys, periods],
                            names=[self.group_key, 'period'])
              .to_frame(index=False)
        )

    def _aggregate_sales(self) -> pd.DataFrame:
        self.trans['period'] = self._to_period(self.trans, self.date_col)
        agg = (
            self.trans
              .groupby([self.group_key, 'period'], as_index=False)
              .agg(
                  demand_qty     = ('quantity','sum'),
                  is_promotion   = ('is_promotion','max'),
                  price          = ('price','mean'),
                  promotion_type = ('promotion_type',
                                     lambda x: x.mode().iat[0]
                                               if not x.mode().empty else 'none'),
                  platform       = ('platform',
                                     lambda x: x.mode().iat[0])
              )
        )
        agg['occurrence'] = (agg['demand_qty'] > 0).astype(int)
        return agg

    def _aggregate_inventory(self) -> pd.DataFrame:
        self.inv['period'] = self._to_period(self.inv, 'date')
        inv_last = (
            self.inv
              .sort_values([self.group_key,'date'])
              .groupby([self.group_key,'period'], as_index=False)
              .last()
              [[self.group_key,'period',
                'stock_level','days_in_stock','restock_quantity']]
        )
        return inv_last

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['month'] = df['period'].dt.month
        df['dow']   = df['period'].dt.dayofweek
        df['woy']   = df['period'].dt.isocalendar().week
        return df

    def build_panel(self) -> pd.DataFrame:
        # Create index with groups and time 
        self._merge_group_key()
        panel = self._build_panel_index()
        sales     = self._aggregate_sales()
        price_avg = (sales.groupby(self.group_key)['price']
                          .mean()
                          .rename('avg_group_price')
                          .reset_index())
        df = (
            panel
            .merge(sales,     on=[self.group_key,'period'], how='left')
            .merge(price_avg, on=self.group_key,             how='left')
            .assign(
                demand_qty     = lambda d: d['demand_qty'].fillna(0),
                occurrence     = lambda d: d['occurrence'].fillna(0),
                is_promotion   = lambda d: d['is_promotion'].fillna(0),
                price          = lambda d: d['price'].fillna(d['avg_group_price']),
                promotion_type = lambda d: d['promotion_type'].fillna('none'),
                platform       = lambda d: d['platform'].fillna('none')
            )
            .drop(columns=['avg_group_price'])
        )
        inv_last = self._aggregate_inventory()
        df = (
            df
            .merge(inv_last, on=[self.group_key,'period'], how='left')
            .fillna({'stock_level':0,'days_in_stock':0,'restock_quantity':0})
        )
        if self.group_key == 'product_id':
            df = df.merge(self.products_df,
                          on='product_id', how='left')
        else:
            static_agg = (
                self.products_df
                    .groupby(self.group_key)
                    .agg(
                        category_id           = ('category_id',           lambda x: x.mode().iat[0]),
                        brand_id              = ('brand_id',              lambda x: x.mode().iat[0]),
                        supplier_id           = ('supplier_id',           lambda x: x.mode().iat[0]),
                        base_cost             = ('base_cost',             'mean'),
                        quality_score         = ('quality_score',         'mean'),
                        avg_competitor_price  = ('avg_competitor_price',  'mean'),
                        is_seasonal           = ('is_seasonal',           'max'),
                        launch_date           = ('launch_date',           'min')
                    )
                    .reset_index()
            )
            df = df.merge(static_agg,
                          on=self.group_key,
                          how='left')
        df = self._add_calendar_features(df)
        self.df_panel = df
        return df

    def split(self, features: list[str] = None, test_split: float = 0.2):
        df = self.df_panel
        if features is None:
            reserved = {self.group_key, 'period', 'demand_qty','occurrence'}
            features = [c for c in df.columns if c not in reserved]
        periods = sorted(df['period'].unique())
        cutoff  = periods[int(len(periods)*(1-test_split))]
        df_tr = df[df['period'] <= cutoff].reset_index(drop=True)
        df_te = df[df['period']  > cutoff].reset_index(drop=True)
        self.df_train, self.df_test = df_tr, df_te
        self.X_tr_cls = df_tr[features]; self.y_tr_cls = df_tr['occurrence']
        self.X_te_cls = df_te[features]; self.y_te_cls = df_te['occurrence']
        pos_tr = df_tr[df_tr['occurrence']==1]
        pos_te = df_te[df_te['occurrence']==1]
        self.X_tr_reg = pos_tr[features]; self.y_tr_reg = pos_tr['demand_qty']
        self.X_te_reg = pos_te[features]; self.y_te_reg = pos_te['demand_qty']
        return (
            self.X_tr_cls, self.X_te_cls,
            self.y_tr_cls, self.y_te_cls,
            self.X_tr_reg, self.X_te_reg,
            self.y_tr_reg, self.y_te_reg
        )

    def fit(self, cv_splits: int = 5):
        num_feats = [c for c in self.X_tr_cls.columns if self.X_tr_cls[c].dtype != 'object']
        cat_feats = [c for c in self.X_tr_cls.columns if c not in num_feats]
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])
        if self.intermittent:
            clf_pipe = Pipeline([('pre', pre), ('clf', LGBMClassifier(class_weight='balanced', random_state=42))])
            clf_params = {
                'clf__num_leaves':      randint(20,150),
                'clf__max_depth':       randint(3,15),
                'clf__learning_rate':   uniform(0.01,0.19),
                'clf__n_estimators':    randint(50,500)
            }
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            self.clf_search = RandomizedSearchCV(clf_pipe, clf_params,
                                                 cv=tscv, scoring='roc_auc',
                                                 n_jobs=-1, random_state=42)
        reg_pipe = Pipeline([('pre', pre), ('reg', LGBMRegressor(objective='poisson', random_state=42))])
        reg_params = {
            'reg__num_leaves':    randint(20,150),
            'reg__max_depth':     randint(3,15),
            'reg__learning_rate': uniform(0.01,0.19),
            'reg__n_estimators':  randint(50,500)
        }
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        self.reg_search = RandomizedSearchCV(reg_pipe, reg_params,
                                            cv=tscv, scoring='neg_root_mean_squared_error',
                                            n_jobs=-1, random_state=42)
        if self.intermittent:
            self.clf_search.fit(self.X_tr_cls, self.y_tr_cls)
        self.reg_search.fit(self.X_tr_reg, self.y_tr_reg)
        return self

    def predict(self, df: pd.DataFrame = None, threshold: float = 0.5) -> pd.DataFrame:
        if df is None:
            X = self.X_te_cls
            base = self.df_test.copy()
        else:
            X = df[self.X_tr_cls.columns]
            base = df.copy()
        if self.intermittent:
            p   = self.clf_search.predict_proba(X)[:, 1]
            occ = (p > threshold).astype(int)
        else:
            p   = np.ones(X.shape[0])
            occ = np.ones(X.shape[0], dtype=int)
        q   = self.reg_search.predict(X)
        base['pred_prob']       = p
        base['pred_occurrence'] = occ
        base['pred_qty']        = q
        base['forecast']        = np.where(occ == 1, q, 0)
        self.results = base
        return base

    def evaluate(self) -> dict:
        metrics = {}
        rmse = np.sqrt(mean_squared_error(self.y_te_reg,
                                          self.results.loc[self.y_te_reg.index,'pred_qty']))
        metrics['rmse'] = rmse
        if self.intermittent:
            y_te = self.y_te_cls
            metrics['roc_auc_or_acc'] = (
                roc_auc_score(y_te, self.results['pred_prob'])
                if y_te.nunique()>1
                else accuracy_score(y_te, self.results['pred_occurrence'])
            )
        return metrics


