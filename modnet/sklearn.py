"""
sklearn API of modnet

This version implements the RR class of the sklearn API
Still TODO: MODNetFeaturizer and MODNet classes

The general pipeline will be:
from sklearn.pipeline import Pipeline
modnet_featurizer = MODNetFeaturizer(...arguments here...)
rr_analysis = RR(...arguments here...)
modnet_model = MODNet(...arguments here...)
p = Pipeline([('featurizer', modnet_featurizer), ('rr', rr_analysis), ('modnet', modnet_model)])

* One note about scikit learn's steps when performing cross-validation:
  A given transformer (e.g. PCA, or RR) will not be executed at each step of the cross validation if you cache it, i.e.
  scikit-learn detects that the inputs are the same (if they are indeed the same) and does not do it several times
  (see https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation).
* Another note about the fit method of a Pipeline object:
  It is possible to pass fit parameters to each step of a Pipeline object
  (see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit)
* Another note about the __init__ method of sklearn derived estimators:
  The method should ONLY set the instance variables. In particular, "every keyword argument accepted by __init__
  should correspond to an attribute on the instance" (as stated in sklearn's developer documentation)
  (see https://scikit-learn.org/stable/developers/develop.html#instantiation).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from typing import Union, Dict
from modnet.preprocessing import (
    get_features_relevance_redundancy,
    get_cross_nmi,
    nmi_target,
    merge_ranked,
)
from modnet.utils import LOG


class MODNetFeaturizer(TransformerMixin, BaseEstimator):
    def __init__(self, matminer_featurizers):
        """Constructor for MODNetFeaturizer"""
        self.matminer_featurizers = matminer_featurizers

    def fit(self, X, y=None):
        """Probably not needed except for some matminer featurizers (e.g. I think SOAP needs to be fitted before)."""
        return self

    def transform(self, X, y=None):
        """Transform the input data (i.e. containing the composition and/or structure to the features"""

    @classmethod
    def from_preset(cls, preset):
        """Initializes a MODNetFeaturizer class based on some preset.

        Args:
            preset: Name of the preset (e.g. "DeBreuck2020" could trigger the Structure+Composition featurizers)

        Notes:
            See in matminer how it is done, e.g. SiteStatsFingerprint.from_preset method.
        """


class RR(TransformerMixin, BaseEstimator):
    """Relevance-Redundancy (RR) feature selection.
    Features are ranked and selected following a relevance-redundancy ratio as developed
    by De Breuck et al. (2020), see https://arxiv.org/abs/2004.14766.

    Use the fit method for computing the most important features.
    Then use the transform method to truncate the input data to those features.

    Attributes:
        n_feat: int
                number of features to keep

        optimal_descriptors: list of length (n_feat)
                            ordered list of best descriptors

    """

    def __init__(
        self,
        n_feat: Union[None, int] = None,
        rr_parameters: Union[None, Dict] = None,
        n_jobs: Union[None, int] = None,
        cross_nmi_kwargs: Union[None, Dict] = None,
        target_nmi_kwargs: Union[None, Dict] = None,
    ):
        """Constructor for RR transformer.

        Args:
            n_feat: Number of features to keep and reorder using the RR procedure (default: None, i.e. all features).
            rr_parameters: Allows tuning of p and c parameters. Currently allows fixing of p and c
             to constant values instead of using the dynamical evaluation. Expects to find keys `"p"` and `"c"`,
              containing either a callable that takes `n` as an argument and returns the desired `p` or `c`,
               or another dictionary containing the key `"value"` that stores a constant value of `p` or `c`.
            n_jobs: max number of processes to use when calculating cross NMI.
            cross_nmi_kwargs: Keyword arguments to be passed down to the modnet.preprocessing.get_cross_nmi
            target_nmi_kwargs: Keyword arguments to be passed down to the modnet.preprocessing.nmi_target
        """
        self.n_feat = n_feat
        self.rr_parameters = rr_parameters
        self.optimal_descriptors = []
        self.optimal_features_by_target = {}
        self.n_jobs = n_jobs
        self.cross_nmi_kwargs = cross_nmi_kwargs if cross_nmi_kwargs is not None else {}
        self.target_nmi_kwargs = (
            target_nmi_kwargs if target_nmi_kwargs is not None else {}
        )

    def fit(self, X, y, nmi_feats_target=None, cross_nmi_feats=None):
        """Ranking of the features. This is based on relevance and redundancy provided as NMI dataframes.
        If not provided (i.e set to None), the NMIs are computed here.
        Nevertheless, it is strongly recommended to compute them separately and store them locally.

        Args:
            X: Training input pandas dataframe of shape (n_samples,n_features)
            y: Training output pandas dataframe of shape (n_samples,n_features)
            nmi_feats_target: NMI between features and targets, pandas dataframe
            cross_nmi_feats: NMI between features, pandas dataframe

        Returns:
            self : object
            Fitted RR transformer
        """

        ranked_lists = []

        if cross_nmi_feats is None:
            cross_nmi_feats = get_cross_nmi(
                X, n_jobs=self.n_jobs, **self.cross_nmi_kwargs
            )

        if nmi_feats_target is None:
            for name in list(y):
                LOG.info(f"Starting NMI computations for target {name}")
                X_temp = X.copy()
                y_temp = y[[name]]

                nmi_feats_target = nmi_target(X_temp, y_temp, **self.target_nmi_kwargs)

                missing = [
                    x for x in nmi_feats_target.index if x not in cross_nmi_feats.index
                ]
                nmi_feats_target = nmi_feats_target.drop(missing, axis=0)
                nmi_feats_target = nmi_feats_target.astype(np.float64)

                rr_results = get_features_relevance_redundancy(
                    nmi_feats_target,
                    cross_nmi_feats,
                    n_feat=self.n_feat,
                    rr_parameters=self.rr_parameters,
                )

                self.optimal_features_by_target[name] = [
                    x["feature"] for x in rr_results
                ]

                ranked_lists.append(self.optimal_features_by_target[name])

        rr_results = get_features_relevance_redundancy(
            nmi_feats_target,
            cross_nmi_feats,
            n_feat=self.n_feat,
            rr_parameters=self.rr_parameters,
        )

        if ranked_lists:
            self.optimal_descriptors = merge_ranked(ranked_lists)
        else:
            self.optimal_descriptors = [x["feature"] for x in rr_results]

        return self

    def transform(self, X, y=None):
        """Transform the inputs X based on a fitted RR analysis. The best n_feat features are kept and returned.

        Args:
            X: input pandas dataframe of shape (n_samples,n_features)
            y: ignored

        Returns:
            X data containing n_feat rows (best features) as a pandas dataframe
        """

        return X[self.optimal_descriptors]


class MODNet(RegressorMixin, BaseEstimator):
    """MODNet model.

    Blabla.

    Notes:
        No assumption on the features here, just a list of numbers.
        What makes a MODNet model special with respect to using directly keras ? I would say that it is always doing
        some joint learning, maybe something else ?
    """

    def __init__(self):
        """Constructor for MODNet model.

        Needs some thinking of what to put in __init__ and fit
        """

    def fit(self, X, y):
        """Fit a MODNet regression model."""

    def predict(self, X):
        """Predict output based on a fitted MODNet regression model"""
