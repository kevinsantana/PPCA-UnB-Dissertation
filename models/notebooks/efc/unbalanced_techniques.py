from enum import Enum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


from constants import LAST_TIME_STEP, LAST_TRAIN_TIME_STEP, EFC_CLF, EFC_HYPER_PARAMS
from shared_functions import custom_confusion_matrix, get_dataset_size, plot_efc_energies, run_elliptic_preprocessing_pipeline, calculate_model_score, train_test_from_splitted, train_test_from_x_y


def efc_baseline(technique:str, fig_folder: str, fig_name: str, only_labeled: bool = True):
    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                                last_time_step=LAST_TIME_STEP,
                                                                                only_labeled=only_labeled)
    clf = EFC_CLF
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(clf, y_test, y_energies, fig_folder, fig_name)
    sizes = get_dataset_size(technique=technique, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    metric_dict = calculate_model_score(technique=technique, y_true=y_test.values, y_pred=y_pred)
    confusion_matrix = custom_confusion_matrix(technique=technique, y_test=y_test, y_pred=y_pred)
    return sizes, metric_dict, confusion_matrix


def equally_distributing_correlating(technique:str, fig_folder: str, fig_name: str, only_labeled: bool = True):
    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                           last_time_step=LAST_TIME_STEP,
                                                                           only_labeled=only_labeled)
    df = train_test_from_splitted(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, return_df=True)
    df = df.sample(frac=1)
    # amount of ilicit classes 4545 rows.
    ilicit_df = df.loc[df['class'] == 1]
    licit_df = df.loc[df['class'] == 0][:4545]
    normal_distributed_df = pd.concat([ilicit_df, licit_df])
    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    new_df.head()
    print('Distribution of the Classes in the subsample dataset')
    print(new_df['class'].value_counts()/len(new_df))
    sns.countplot(x='class', data=new_df, palette=['red', 'blue'])
    plt.title('Equally Distributed Classes', fontsize=14)
    plt.show()
    X = new_df.drop(['class'], axis=1)
    y = new_df['class']
    X_train, X_test, y_train, y_test = train_test_from_x_y(X, y)
    clf = EFC_CLF
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(clf, y_test, y_energies, fig_folder, fig_name)
    sizes = get_dataset_size(technique=technique, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    metric_dict = calculate_model_score(technique=technique, y_true=y_test.values, y_pred=y_pred)
    confusion_matrix = custom_confusion_matrix(technique=technique, y_test=y_test, y_pred=y_pred)
    return sizes, metric_dict, confusion_matrix


def smote(technique:str, fig_folder: str, fig_name: str, only_labeled: bool = True):
    X, y = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                               last_time_step=LAST_TIME_STEP,
                                               only_labeled=only_labeled,
                                               only_x_y=True)
    smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
    X_sm, y_sm = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_from_x_y(X_sm, y_sm)
    clf = EFC_CLF
    clf.fit(X_train, y_train, base_class=0)
    y_pred, y_energies = clf.predict(X_test, return_energies=True)
    plot_efc_energies(clf, y_test, y_energies, fig_folder, fig_name)
    sizes = get_dataset_size(technique=technique, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    metric_dict = calculate_model_score(technique=technique, y_true=y_test.values, y_pred=y_pred)
    confusion_matrix = custom_confusion_matrix(technique=technique, y_test=y_test, y_pred=y_pred)
    return sizes, metric_dict, confusion_matrix


def grid_search(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline, kf: StratifiedKFold):
    grid = GridSearchCV(pipeline, param_grid=EFC_HYPER_PARAMS, cv=kf, scoring='recall', return_train_score=True)
    grid.fit(X_train, y_train)
    print('Best parameters:', grid.best_params_)
    print('Best score:', grid.best_score_)
    y_pred = grid.predict(X_test)
    return y_pred


class ResamplingEnum(Enum):
    OVER_SAMPLER = "RandomOverSampler"
    UNDER_SAMPLER = "RandomUnderSampler"


def random_resampling(technique:str,
                      fig_folder: str,
                      fig_name: str,
                      resampling: ResamplingEnum,
                      only_labeled: bool = True):
    sampling = None
    if resampling == ResamplingEnum.OVER_SAMPLER.value:
        sampling = RandomOverSampler(random_state=139)
    elif resampling == ResamplingEnum.UNDER_SAMPLER.value:
        sampling = RandomUnderSampler(random_state=139)

    X_train, X_test, y_train, y_test = run_elliptic_preprocessing_pipeline(last_train_time_step=LAST_TRAIN_TIME_STEP,
                                                                           last_time_step=LAST_TIME_STEP,
                                                                           only_labeled=only_labeled)
    X_resampled, y_resampled = sampling.fit_resample(X_train, y_train)
    
    print('Genuine:', y_resampled.value_counts()[0], '/', round(y_resampled.value_counts()[0]/len(y_resampled) * 100,2), '% of the dataset')
    print('Frauds:', y_resampled.value_counts()[1], '/',round(y_resampled.value_counts()[1]/len(y_resampled) * 100,2), '% of the dataset')
    
    clf = EFC_CLF
    random_overs_pipeline = make_pipeline(sampling, clf)    
    random_overs_pipeline.fit(X_resampled, y_resampled)
    y_pred = random_overs_pipeline.predict(X_test)

    # plot_efc_energies(random_overs_pipeline, y_test, y_energies, fig_folder, fig_name)

    sizes = get_dataset_size(technique=technique, X_train=X_resampled, X_test=X_test, y_train=y_resampled, y_test=y_test)
    metric_dict = calculate_model_score(technique=technique, y_true=y_test.values, y_pred=y_pred)
    confusion_matrix = custom_confusion_matrix(technique=technique, y_test=y_test, y_pred=y_pred)
    return sizes, metric_dict, confusion_matrix

