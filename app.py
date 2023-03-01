import numpy as np
import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes

from utils.functions import *

from utils.ui import (
    dataset_selector,
    generate_snippet,
    polynomial_degree_selector,
    introduction,
    model_selector,
    upload_data,
    generate_data_snippet,
)

st.set_page_config(
    page_title="Playground", layout="wide", page_icon="./images/flask.png"
)


def sidebar_upload_controllers():
    data_set = upload_data()
    model_type, model = model_selector()
    return (
        data_set,
        model_type,
        model,
    )


def sidebar_controllers():
    dataset, n_samples, train_noise, test_noise, n_classes = dataset_selector()
    model_type, model = model_selector()
    x_train, y_train, x_test, y_test = generate_data(
        dataset, n_samples, train_noise, test_noise, n_classes
    )
    st.sidebar.header("Feature engineering")
    degree = polynomial_degree_selector()

    return (
        dataset,
        n_classes,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        degree,
        train_noise,
        test_noise,
        n_samples,
    )


def body(
    x_train, x_test, y_train, y_test, degree, model, model_type, train_noise, test_noise
):
    introduction()
    col1, col2 = st.columns((1, 1))

    with col1:
        plot_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    x_train, x_test = add_polynomial_features(x_train, x_test, degree)
    model_url = get_model_url(model_type)

    (
        model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
        duration,
    ) = train_model(model, x_train, y_train, x_test, y_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }
    # snippet_par = {
    #     "model": model, "model_type": model_type, 'n_samples': n_samples , 'train_noise': train_noise, 'test_noise': test_noise,
    #     'dataset': dataset, "degree": degree
    # }

    snippet = generate_snippet(
        model, model_type, n_samples, train_noise, test_noise, dataset, degree
    )

    model_tips = get_model_tips(model_type)

    fig = plot_decision_boundary_and_metrics(
        model, x_train, y_train, x_test, y_test, metrics
    )

    plot_placeholder.plotly_chart(fig, True)
    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    code_header_placeholder.header("**Retrain the same model in Python**")
    snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


def uplouded_data_body(
    x_train, x_test, y_train, y_test, model, model_type, dataset
):
    st.markdown('** Data splits**')
    col1, col2, col3 = st.columns((1, 1, 2))

    with col1:
        plot_placeholder = st.empty()
        st.write('Training set')
        train_set = st.empty()
        st.write('train_accuracy')
        train_placeholder = st.empty()
        st.write('test_accuracy')
        test_placeholder = st.empty()

    with col2:
        st.write('Test set')
        test_set = st.empty()
        st.write('train_f1')
        train_f1_placeholder = st.empty()
        st.write('test_f1')
        test_f1_placeholder = st.empty()
    with col3:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    model_url = get_model_url(model_type)
    (
        model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
        duration,
    ) = train_model(model, x_train, y_train, x_test, y_test)
    snippet = generate_data_snippet(
        model, model_type, dataset
    )
    train_placeholder.info(train_accuracy)
    test_placeholder.info(test_accuracy)
    train_f1_placeholder.info(train_f1)
    test_f1_placeholder.info(test_f1)

    st.write('Model Parameters')
    st.write(model.get_params())

    model_tips = get_model_tips(model_type)

    train_set.info(X_train.shape)
    test_set.info(X_test.shape)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


def boot_body(model_type, train_accuracy, test_accuracy, train_f1, test_f1, duration):
    col1, col2, col3 = st.columns((1, 1, 2))
    with col1:
        plot_placeholder = st.empty()
        st.write('train_accuracy')
        train_placeholder = st.empty()
        st.write('test_accuracy')
        test_placeholder = st.empty()

    with col2:
        st.write('train_f1')
        train_f1_placeholder = st.empty()
        st.write('test_f1')
        test_f1_placeholder = st.empty()
    with col3:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    if model_type == 'Random Forest ':
        model_type = 'Random Forest'
    model_url = get_model_url(model_type)
    train_placeholder.info(train_accuracy)
    test_placeholder.info(test_accuracy)
    train_f1_placeholder.info(train_f1)
    test_f1_placeholder.info(test_f1)
    model_tips = get_model_tips(model_type)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    # snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


if __name__ == "__main__":
    choise_data = st.sidebar.radio(
        "choise your Data", ('Make your Data', 'Upload your CSV file', 'Bot'))
    if choise_data == 'Make your Data':
        (
            dataset,
            n_classes,
            model_type,
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            degree,
            train_noise,
            test_noise,
            n_samples,
        ) = sidebar_controllers()
        body(
            x_train,
            x_test,
            y_train,
            y_test,
            degree,
            model,
            model_type,
            train_noise,
            test_noise,
        )
    elif choise_data == 'Bot':
        data_set = upload_data()
        if data_set is not None:
            selected_var = []
            t = []
            df = pd.read_csv(data_set)
            st.subheader(' Glimpse of dataset')
            st.write(df.head())
            size = data_set.getvalue()
            df_size = len(size)
            (n_rows, n_coloumn, n_class) = data_shape(df)
            (df, std, mean) = pre_proses_data(df)
            submit = st.sidebar.button("submit")
            if submit == True:
                st.write("Data Set After preprossing ")
                st.write(df.head())
                same_id = sim_id(n_rows, n_coloumn, n_class,
                                 df_size, std, mean)
                st.write(same_id)
                for i in same_id:
                    data_ref = pd.read_csv(str(i)+'.csv', header=None)
                    data_ref = data_ref.iloc[1:11, :]
                    models = data_ref.to_numpy()
                    selected_var.append(best_data(models, df))
                selected_var = sorted(
                    selected_var, key=lambda x: x[16], reverse=True)
                st.dataframe(selected_var)
                t = selected_var[0]
                (model, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors,
                 metric, solver, penalty, C, max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, duration) = t
                a = []
                temp = {}
                array = [model, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors,
                         metric, solver, penalty, C, max_iter, kernel]
                b = ['model', 'criterion', 'max_depth', 'min_samples_split', 'max_features', 'learning_rate', 'n_estimators',
                     'n_neighbors', 'metric', 'solver', 'penalty', ' C', 'max_iter', 'kernel']
                a.append(b)
                a.append(array)
                data = dict(zip(b, t))
                for x, y in data.items():
                    if y != '0' and y != '0.0':
                        temp[x] = y

                st.table(temp)
                boot_body(model, train_accuracy, test_accuracy,
                          train_f1, test_f1, duration)

        else:
            st.info('Awaiting for CSV file to be uploaded.')
    else:
        (
            data_set,
            model_type,
            model,

        ) = sidebar_upload_controllers()

        if data_set is not None:
            df = pd.read_csv(data_set)
            st.subheader(' Glimpse of dataset')
            st.write(df.head())
            df = encodeing_df(df)
            df = replace_null(df)
            df = scaling(df)
            st.write("Data Set After preprossing ")
            st.write(df.head())
            with st.sidebar.header('2. Set Parameters'):
                split_size = st.sidebar.slider(
                    'Data split ratio (% for Training Set)', 10, 90, 80, 5)
            submit = st.sidebar.button("submit")
            if submit == True:

                (
                    X_train, X_test, Y_train, Y_test
                ) = split_data(df, split_size)
                uplouded_data_body(X_train, X_test, Y_train,
                                   Y_test, model, model_type, df)
        else:
            st.info('Awaiting for CSV file to be uploaded.')
