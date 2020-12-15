"""Train the baseline model i.e. a logistic regression on the average of the resnet features and
and make a prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=".\\data", type=Path,
                    help="directory where data is stored")
parser.add_argument("--num_runs", default=5, type=int,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")
def get_tile_number(s):
    nb = s[:3]
    if s[1]=="_":
        nb = s[:1]
    if s[2]=="_":
        nb = s[:2]
        
    return nb

def get_annotations_ID_Tiles():
    annotations = pd.read_csv(".\\data\\train_input\\train_tile_annotations.csv")
    
    ID = []
    TileID = []
    y = [] 
    for i in range(len(annotations)):
        name = annotations["TileName"][i]
        
        ID.append(name[3:6])
        
        TileID.append(get_tile_number(name[3:6]+name[22:25]))
        
        y.append(int(annotations["Target"][i]))

    
    return ID,TileID,y

def get_filenames(ID,train_dir= ".\\data\\train_input\\resnet_features"):       
    liste_ID = []
    nID = ID[0]
    for i in range(len(ID)):
        if ID[i] != nID:
            liste_ID.append(nID)
            nID = ID[i]
    liste_ID.append(nID)
    filenames_train = [train_dir +"\\ID_{}_annotated.npy".format(idx) for idx in liste_ID]
    return filenames_train

    
    
    
def get_features(filenames):
    features = []
    for f in filenames :
        features.append(np.load(f))
    return features

def reshape_features(features):
    reshaped = []
    for feat in features :
        for i in range(len(feat)):
            reshaped.append(feat[i][3:])
    return reshaped

        


def get_average_features(filenames):
    """Load and aggregate the resnet features by the average.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 2048)`
    """
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f)

        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]

        aggregated_features = np.mean(patient_features, axis=0)
        features.append(aggregated_features)

    features = np.stack(features, axis=0)
    return features


if __name__ == "__main__":
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input"  / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)
    
    # Get the filenames for train
    filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    # Get the labels
    labels_train = train_output["Target"].values

    assert len(filenames_train) == len(labels_train)


    # Get the numpy filenames for test
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), filename
    ids_test = [f.stem for f in filenames_test]


    # Get the resnet features and aggregate them by the average
    features_train = get_average_features(filenames_train)
    features_test = get_average_features(filenames_test)

    # -------------------------------------------------------------------------
    # Use the average resnet features to predict the labels

    # Multiple cross validations on the training set
    aucs = []
    for seed in range(args.num_runs):
        # Use logistic regression with L2 penalty
        estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")

        cv = sklearn.model_selection.StratifiedKFold(n_splits=args.num_splits, shuffle=True,
                                                     random_state=seed)

        # Cross validation on the training set
        auc = sklearn.model_selection.cross_val_score(estimator, X=features_train, y=labels_train,
                                                      cv=cv, scoring="roc_auc", verbose=0)

        aucs.append(auc)

    aucs = np.array(aucs)

    print("Predicting weak labels by mean resnet")
    print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))


    # -------------------------------------------------------------------------
    # Prediction on the test set

    # Train a final model on the full training set
    estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    estimator.fit(features_train, labels_train)

    preds_test = estimator.predict_proba(features_test)[:, 1]

    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.data_dir / "preds_test_baseline.csv")

#Apprentissage
args = parser.parse_args()

# -------------------------------------------------------------------------
# Load the data
assert args.data_dir.is_dir()

train_dir = args.data_dir / "train_input" / "resnet_features"
test_dir = args.data_dir / "test_input"  / "resnet_features"

train_output_filename = args.data_dir / "train_output.csv"

train_output = pd.read_csv(train_output_filename)


  
ID,TileID,y = get_annotations_ID_Tiles()
filenames_train = get_filenames(ID)

x_train = get_features(filenames_train)
x_train = reshape_features(x_train)

estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.01, solver="liblinear")
estimator.fit(x_train, y)

seuil_proba = 0.5
seuil_metastase = 1

filenames_test = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]

results = []
for f in filenames_test:
    IDtest = np.load(f)
    x_test = [IDtest[i][3:] for i in range(len(IDtest))]
    pred = estimator.predict_proba(x_test)
    results.append(np.quantile(pred[:,1],0.995) )

y_predicted = (np.array(results) >=0.5)
y_true = np.array(labels_train)
sum(y_predicted == y_true)/len(y_true)

filenames_test2 = sorted(test_dir.glob("*.npy"))


for f in filenames_test2:
    IDtest = np.load(f)
    x_test = [IDtest[i][3:] for i in range(len(IDtest))]
    pred = estimator.predict_proba(x_test)
    results.append(np.quantile(pred[:,1],0.99) )

y_predicted = (np.array(results) >=0.5)

test_output = pd.DataFrame({"ID": ids_number_test, "Target": results})
test_output.set_index("ID", inplace=True)
test_output.to_csv(args.data_dir / "Test1.csv")

file = pd.read_csv("preds_test_baseline.csv")
file["Target"]=results
test_output.set_index("ID", inplace=True)
file.to_csv(args.data_dir / "Test1.csv")
