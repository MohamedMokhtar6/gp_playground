from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from models.utils import model_infos, model_urls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models_2.NaiveBayes import nb_param_selector
from models_2.RandomForet import rf_param_selector
from models_2.DecisionTree import dt_param_selector
from models_2.LogisticRegression import lr_param_selector
from models_2.KNearesNeighbors import knn_param_selector
from models_2.SVC import svc_param_selector
from models_2.GradientBoosting import gb_param_selector


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def plot_decision_boundary_and_metrics(
    model, x_train, y_train, x_test, y_test, metrics
):
    d = x_train.shape[1]

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p)
                   for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [
            {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Decision Boundary", None, None),
        row_heights=[0.7, 0.30],
    )

    heatmap = go.Heatmap(
        x=xx[0],
        y=y_,
        z=Z,
        colorscale=["tomato", "rgb(27,158,119)"],
        showscale=False,
    )

    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(heatmap, row=1, col=1,).add_trace(train_data).add_trace(
        test_data
    ).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2"
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_accuracy"],
            title={"text": f"Accuracy (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_accuracy"]},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_f1"],
            title={"text": f"F1 score (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_f1"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig


def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test


def build_model(df, model):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    st.markdown('** Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('** Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    model.fit(X_train, Y_train)
    st.write('Model Info')
    st.info(model)

    st.markdown('** Training set**')
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = np.round(accuracy_score(Y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(Y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(Y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(Y_test, y_test_pred, average="weighted"), 3)
    st.write('train_accuracy')
    st.info(train_accuracy)
    st.write('train_f1')
    st.info(train_f1)
    st.write('test_accuracy')
    st.info(test_accuracy)
    st.write('test_f1')
    st.info(test_f1)

    st.write('Model Parameters')
    st.write(model.get_params())


def split_data(df, split_size):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    return X_train, X_test, Y_train, Y_test


def encodeing_df(df):
    col_name = []
    label_encoder = LabelEncoder()
    for (colname, colval) in df.iteritems():
        if colval.dtype == 'object':
            col_name.append(colname)
    for col in col_name:
        df[col] = label_encoder.fit_transform(df[col])
    return df


def replace_null(df):
    col_nan = []
    for (colname, colval) in df.iteritems():
        if df[colname].isnull().values.any() == True:
            col_nan.append(colname)

    for col in col_nan:
        mean_value = df[col].mean()
        df[col].fillna(value=mean_value, inplace=True)
    return df


def scaling(df):
    x = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last
    m = x.max() - x.min()
    m = m.to_dict()
    l = []
    for key, val in m.items():
        if val == 0:
            l.append(key)
    for i in l:
        x = x.drop([i], axis=1)

    df_norm = (x-x.min())/(x.max()-x.min())
    df_norm = pd.concat((df_norm, y), 1)

    return df_norm


def cal_mean(df):
    df_mean = df.mean(axis=0).mean(axis=0)
    return df_mean


def cal_std(df):
    col_std = df.std()
    col = col_std.mean(axis=0)
    return col


def data_shape(df):
    # n_class	n_rows	n_coloumn	data_size
    shape = df.shape
    n_rows = shape[0]
    n_coloumn = shape[1]
    index_last_col = df.columns[-1]
    n_class = len(df[index_last_col].unique())
    return n_rows, n_coloumn, n_class


def pre_proses_data(df):
    df = encodeing_df(df)
    df = replace_null(df)
    std = cal_std(df)
    std = np.round(std, 5)
    mean = cal_mean(df)
    mean = np.round(mean, 5)
    df = scaling(df)
    return df, std, mean


def sim_id(n_rows, n_coloumn, n_class, df_size, std, mean):
    dataf = pd.read_csv('datasets.csv', header=None)
    datasets = dataf.to_numpy()
    max_id = 0
    max_score = 0
    for dataset in datasets:
        score = 0
        (id, num_class, num_row, nm_col, dataSiz, data_mean, data_std) = dataset
        nclass = (n_class/num_class)
        nrows = (n_rows/num_row)
        ncol = (n_coloumn/nm_col)
        siz = (df_size/dataSiz)
        meannn = (mean/data_mean)
        stddd = (std/data_std)
        if (nclass > 0.6 and nclass < 1.4):
            score += 1
        if (nrows > 0.6 and nrows < 1.4):
            score += 1
        if (ncol > 0.6 and ncol < 1.4):
            score += 1
        if (siz > 0.6 and siz < 1.4):
            score += 1
        if (meannn > 0.6 and meannn < 1.4):
            score += 1
        if (stddd > 0.6 and stddd < 1.4):
            score += 1
        if max_score >= score:
            max_score = max_score
            max_id = max_id
        else:
            max_score = score
            max_id = id
    return int(max_id)


def best_data(models, new_data):
    a = []
    for model in models:
        (model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
         max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, duration, data_id, n_class, n_rows, n_coloumn, data_size, mean, std) = model
        if model_name == 'Random Forest ' or model_name == 'Random Forest':
            model = rf_param_selector(
                criterion, int(n_estimators), int(max_depth), int(min_samples_split), max_features)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1,
             duration) = train_model(model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Decision Tree':
            if max_features != 'sqrt' or max_features != 'log2':
                max_features = None
            model = dt_param_selector(criterion, int(
                max_depth), int(min_samples_split), max_features)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
        elif model_name == 'SVC':
            model = svc_param_selector(float(C), kernel)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'K Nearest Neighbors':
            model = knn_param_selector(int(n_neighbors), metric)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Gradient Boosting':
            model = gb_param_selector(
                float(learning_rate), int(n_estimators), int(max_depth))
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Logistic Regression':
            model = lr_param_selector(solver, penalty, float(C), int(max_iter))
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
        elif model_name == 'NaiveBayes':
            model = nb_param_selector()
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
    a = sorted(a, key=lambda x: x[16], reverse=True)
    return a[0]
