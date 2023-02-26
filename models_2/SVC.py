from sklearn import svm
from sklearn.svm import SVC


def svc_param_selector(C,kernel):

    #  st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
    params = {"C": C, "kernel": kernel}
    model = SVC(**params)
    return model
