import tensorflow as tf
import argparse
from contextlib import redirect_stdout
from model import get_model
from convert import load_dataset
from config import NUM_EPOCHS, BASE_RESULT_DIR, BASE_GENERATED_IMAGE_DIR, BASE_SAVE_MODEL_DIR


if __name__ == '__main__':
    desc = 'Train the model using specific dataset directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--program', choices=('v1', 'v2'), required=True, help='convert program type')
    parser.add_argument('-f', '--filter_column', default=None, help='filter query for dataset columns')
    parser.add_argument('-t', '--threshold', type=float, help='threshold value for correlation')
    args = parser.parse_args()

    if args.filter_column:
        filter_column = args.filter_column
    else:
        filter_column = 'all'
    result_filename = f'{BASE_RESULT_DIR}/{args.program}_{filter_column}.csv'
    train_chkpth_dir = f'{BASE_SAVE_MODEL_DIR}/{args.program}_{filter_column}'
    dataset_directory = f'{BASE_GENERATED_IMAGE_DIR}/{args.program}/{filter_column}'
    if args.program == 'v2':
        result_filename = f'{result_filename[:-4]}_{args.threshold}.csv'
        train_chkpth_dir = f'{train_chkpth_dir}_{args.threshold}'
        dataset_directory = f'{dataset_directory}/{args.threshold}'            
    
    train_ds, validation_ds = load_dataset(
        class_name=['1.benign', '1.mirai.ack'],
        dataset_directory=dataset_directory
    )

    chkpth_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='%s/checkpoint_{epoch:03d}' % train_chkpth_dir,
        verbose=1,
        save_best_only=True,
    )

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives()
    ]
    model = get_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=validation_ds,
        callbacks=[tf.keras.callbacks.CSVLogger(result_filename), chkpth_callback],
        verbose=2,
    )
