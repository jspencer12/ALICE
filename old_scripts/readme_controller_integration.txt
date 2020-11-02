Step 1: Set up your test folder somewhere

    > mkdir -p ~/models/pacifica_longitudinal_JStest1

Step 2: Copy config from another model

    This shows you where some of the other models live:
    > gsutil ls gs://aurora-control-models-prod/PACIFICA_V1/longitudinal/pacifica-all/**

    We'll choose the most recent
    > bazel run //control/ml/tools:extract_model_config -- -i gs://aurora-control-models-prod/PACIFICA_V1/longitudinal/pacifica-all/2020_03_11_05_57_49/info -o ~/models/pacifica_longitudinal_JStest1

Step 3: Build train/test dataset

    > bazel run //control/ml:training_data_generator -- -f ~/models/pacifica_longitudinal_JStest1
    (Might need the --cloud_read flag)

Step 4: Make (Copy) a TF model

    > mkdir ~/models/pacifica_longitudinal_JStest1/tf_model/
    > cp ~/av/controls/ml/learned_controller_model/pacifica_longitudinal_model.py ~/models/pacifica_longitudinal_JStest1/models/
