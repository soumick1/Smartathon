import sagemaker
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import boto3
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.session import s3_input

role = get_execution_role()
region = boto3.Session().region_name
bucket = '________' # Please specify the bucket where the SVS images are downloaded
sagemaker_session = sagemaker.Session()
s3 = boto3.resource('s3', region_name=region)


def generate_tf_records(base_folder, input_files, output_file, n_image, slide=None):
    record_file = output_file

    count = n_image
    with tf.io.TFRecordWriter(record_file) as writer:
        while count:
            filename, label = random.choice(input_files)
            temp_img = plt.imread(os.path.join(base_folder, filename))
            count -= 1

            image_string = np.float32(temp_img).tobytes()
            slide_string = slide.encode('utf-8') if slide else None
            tf_example = image_example(image_string, label, slide_string)
            writer.write(tf_example.SerializeToString())

processor = Processor(image_uri=image_name,
                        role=get_execution_role(),
                        instance_count=16,  # run the job on 16 instances
                        base_job_name='processing-base',  # should be unique name
                        instance_type='ml.m5.4xlarge',
                        volume_size_in_gb=1000)

processor.run(inputs=[ProcessingInput(
    source=f's3://<bucket_name>/_____',  # s3 input prefix
    s3_data_type='S3Prefix',
    s3_input_mode='File',
    s3_data_distribution_type='ShardedByS3Key',  # Split the data across instances
    destination='/opt/ml/processing/input')],  # local path on the container
    outputs=[ProcessingOutput(
        source='/opt/ml/processing/output',  # local output path on the container
        destination=f's3://<bucket_name>/____/'  # output s3 location
    )],
    arguments=['10000'],  # number of tiled images per TF record for training dataset
    wait=True,
    logs=True)


HEIGHT = 512
WIDTH = 512
DEPTH = 3
NUM_CLASSES = 3


def dataset_parser(value):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'slide_string': tf.io.FixedLenFeature([], tf.string)
    }
    record = tf.io.parse_single_example(value, image_feature_description)
    image = tf.io.decode_raw(record['image_raw'], tf.float32)
    image = tf.cast(image, tf.float32)
    image.set_shape([DEPTH * HEIGHT * WIDTH])
    image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
    label = tf.cast(record['label'], tf.int32)
    slide = record['slide_string']

    return image, label, slide





key = '____________'

file = [f for f in s3.Bucket(bucket).objects.filter(Prefix=key).limit(1)][0]
local_file = file.key.split('/')[-1]
s3.Bucket(bucket).download_file(file.key, f'./images/{local_file}')

raw_image_dataset = tf.data.TFRecordDataset(f'./images/{local_file}')
parsed_image_dataset = raw_image_dataset.map(dataset_parser)


train_instance_type='ml.p3.8xlarge'
train_instance_count = 4
gpus_per_host = 4
num_of_shards = gpus_per_host * train_instance_count

distributions = {'mpi': {
    'enabled': True,
    'processes_per_host': gpus_per_host
    }
}





# Sharding
client = boto3.client('s3')
result = client.list_objects(Bucket=bucket, Prefix='_____/train/', Delimiter='/')

j = -1
for i in range(num_of_shards):
    copy_source = {
        'Bucket': bucket,
        'Key': result['Contents'][i]['Key']
     }
    print(result['Contents'][i]['Key'])
    if i % gpus_per_host == 0:
        j += 1
    dest = '_______/train_sharded/' + str(j) +'/' + result['Contents'][i]['Key'].split('/')[2]
    print(dest)
    s3.meta.client.copy(copy_source, bucket, dest)


svs_tf_sharded = f's3://{bucket}/tcga-svs-tfrecords'
shuffle_config = sagemaker.session.ShuffleConfig(234)
train_s3_uri_prefix = svs_tf_sharded
remote_inputs = {}

for idx in range(gpus_per_host):
    train_s3_uri = f'{train_s3_uri_prefix}/train_sharded/{idx}/'
    train_s3_input = s3_input(train_s3_uri, distribution ='ShardedByS3Key', shuffle_config=shuffle_config)
    remote_inputs[f'train_{idx}'] = train_s3_input
    remote_inputs['valid_{}'.format(idx)] = '{}/valid'.format(svs_tf_sharded)
remote_inputs['test'] = '{}/test'.format(svs_tf_sharded)



#TRAINING
local_hyperparameters = {'epochs': 5, 'batch-size' : 16, 'num-train':160000, 'num-val':8192, 'num-test':8192}

estimator_dist = TensorFlow(base_job_name='svs-horovod-cloud-pipe',
                            entry_point='src/train.py',
                            role=role,
                            framework_version='2.1.0',
                            py_version='py3',
                            distribution=distributions,
                            volume_size=1024,
                            hyperparameters=local_hyperparameters,
                            output_path=f's3://{bucket}/output/',
                            instance_count=4,
                            instance_type=train_instance_type,
                            input_mode='Pipe')

estimator_dist.fit(remote_inputs, wait=True)

model_data = f's3://{bucket}/output/{estimator_dist.latest_training_job.name}/output/model.tar.gz'

model = TensorFlowModel(model_data=model_data,
                        role=role, framework_version='2.1.0')

predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
