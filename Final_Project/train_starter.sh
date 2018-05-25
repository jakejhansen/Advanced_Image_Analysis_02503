(cd ~/dev/models/research export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim)

python3 xml_to_csv.py

python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

cp data/test.record object_detection/data &
cp data/train.record object_detection/data 

cd object_detection

tensorboard --logdir=training/ --port=8009 &

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config