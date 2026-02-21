from train.train_util import load_training_dataset
from train.train_cnn import build_cnn,train_cnn, plot_results, compile_model
from evaluate.evaluate import evaluate_cnn, plot_cnn_confusion_matrix,plot_cnn_summary,plot_cnn_confusion_matrix,plot_cnn_roc_curve

from train.train_util import save_model, load_test_dataset

from evaluate.experiment_tracking import init_mlflow, mlflow_end_run,mlflow_eda, mlflow_cnn

train_ds,val_ds=load_training_dataset()

cnn_model=build_cnn()
compile_model(cnn_model)

history=train_cnn(model=cnn_model,train_ds=train_ds,val_ds=val_ds)

plot_results(history)

test_ds= load_test_dataset()

y_true, y_probs, y_pred=evaluate_cnn(model=cnn_model,test_ds=test_ds)

plot_cnn_confusion_matrix(y_true, y_pred)

roc_auc=plot_cnn_roc_curve(y_true,y_probs)

metrics_values=plot_cnn_summary(y_true, y_pred, roc_auc)

save_model("cnn_model.pkl",cnn_model)
print("CNN model saved")


init_mlflow()
mlflow_end_run()
mlflow_cnn(cnn_model=cnn_model,metrics=metrics_values)

mlflow_eda()

mlflow_end_run()