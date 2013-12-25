"""Search over specified parameter in parallel on IPython
"""
import os
import numpy as np
from IPython.parallel import Client
from sklearn.externals import joblib
from sklearn.cross_validation import check_cv
from sklearn.base import is_classifier, clone
from sklearn.utils.validation import _num_samples, check_arrays
from sklearn.grid_search import ParameterGrid, BaseSearchCV, _check_param_grid

DATA_FILENAME_TEMPLATE = 'data_%03d.npy'

class OnProgressError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def evaluate(estimator, data_filename, params):
        from sklearn.externals import joblib
        X_train, y_train, X_test, y_test = joblib.load(data_filename, mmap_mode='c')
        estimator.set_params(**params)
        estimator.fit(X_train, y_train)
        test_score = estimator.score(X_test, y_test)
        return test_score

class IPyGridSearchCV(BaseSearchCV):
    """Search over specified parameter in parallel on IPython
    """
    def __init__(self, estimator, param_grid, dataset_filenames=None,
                 sync=True, scoring=None,
                 loss_func=None, score_func=None, fit_params=None, n_jobs=1, iid=True,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs'):
        super(IPyGridSearchCV, self).__init__(
            estimator, scoring, loss_func, score_func, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch)
        self.param_grid = param_grid
        self.dataset_filenames = dataset_filenames
        self.sync = sync
        _check_param_grid(param_grid)

    def fit(self, X, y=None, **params):
        return self._fit(X, y, ParameterGrid(self.param_grid))

    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        estimator = self.estimator
        cv = self.cv

        n_samples = _num_samples(X)

        X, y = check_arrays(X, y, allow_lists=True, sparse_format='csr')
        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)

        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
        if not self.dataset_filenames:
            self.save_dataset_filename(X, y, cv)

        dataset_filenames = self.dataset_filenames

        client = Client()
        lb_view = client.load_balanced_view()

        if self.verbose > 0:
            print("Number of CPU core %d" % len(client.ids()))

        self.tasks = [([lb_view.apply(evaluate, estimator, dataset_filename, params)
                        for dataset_filename in dataset_filenames], params)
                            for params in parameter_iterable]
        if self.sync:
            self.wait()
            self.set_grid_scores()
            self.set_best_score_params()

            if self.refit:
                self.set_best_estimator(estimator)
        return self


    def save_dataset_filename(self, X, y, cv):
        dataset_filenames = []
        for i, (train, test) in enumerate(cv):
            cv_fold = ([X[k] for k in train], y[train], [X[k] for k in test], y[test])
            cv_split_filename = get_fullpath(DATA_FILENAME_TEMPLATE % i)
            joblib.dump(cv_fold, cv_split_filename)
            dataset_filenames.append(cv_split_filename)
        self.dataset_filenames = dataset_filenames

    def wait(self):
        return [task.get() for task_group in self.tasks
                                    for task in task_group[0]]

    def get_progress(self):
        return np.mean([task.ready() for task_group in self.tasks
                                                for task in task_group[0]])

    def set_grid_scores(self):
        if self.get_progress() != 1:
            raise OnProgressError('On process')
        grid_scores = []
        for task in self.tasks:
            grid_scores.append((np.mean([t.get() for t in task[0]]), task[1]))
        self.grid_scores_ = grid_scores

    def set_best_score_params(self):
        best = sorted(self.grid_scores_, reverse=True)[0]
        self.best_score_ = best[0]
        self.best_params_ = best[1]

    def set_best_estimator(self, estimator):
        best_estimator = clone(estimator).set_params(
                **self.best_params_)
        if y is not None:
            best_estimator.fit(X, y, **self.fit_params)
        else:
            best_estimator.fit(X, **self.fit_params)
        self.best_estimator_ = best_estimator

def create_clf():
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    clf = Pipeline([
        ('vect', TfidfVectorizer(
                    token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
        )),
        ('svc', SVC()),
    ])
    return clf

if __name__ == '__main__':
    from sklearn.datasets import load_iris, fetch_20newsgroups
    from sklearn.cross_validation import KFold
    news = fetch_20newsgroups(subset='all')
    n_samples = 3000
    X, y = news.data[:n_samples], news.target[:n_samples]
    cv = KFold(len(X), 4, shuffle=True, random_state=0)
    params = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

    estimator = create_clf()
    ipy_gridsearchcv = IPyGridSearchCV(estimator, params, cv=cv)
    ipy_gridsearchcv.fit(X, y)
    print ipy_gridsearchcv.best_params_, ipy_gridsearchcv.best_score_
