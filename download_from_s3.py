# TODO: update requirements.txt
import boto3
import botocore
import os

# Create ./aws_creds.txt file
# with aws s3 access_id in first line
# and access_key in second line
creds = [line.strip().decode("utf-8") for line in open("aws_creds.txt", "rb")]
ACCESS_ID = creds[0]
ACCESS_KEY = creds[1]

BUCKET_NAME = 'aslive-ml-models'

s3 = boto3.resource('s3',
                    aws_access_key_id=ACCESS_ID,
                    aws_secret_access_key=ACCESS_KEY)

model_s3_root = 'models/'

for s3_object in s3.Bucket(BUCKET_NAME).objects.all():

    path, filename = os.path.split(s3_object.key)
    print("path: " + path)
    print("filename: " + filename + "\n")

    if (path.startswith(model_s3_root)) and (filename != ''):
        print("match\n")  # We have a file to download

        new_dir = path.replace(model_s3_root, '', 1)  # Strips out 'models/' from the path since this isn't a local dir

        print('new_dir: ' + new_dir)

        if not os.path.exists(new_dir):  # We need to check if the os directories are in place to download it
            os.makedirs(new_dir)  # If not, we need to create the directory (or directories) for that file
        try:
            s3.Bucket(BUCKET_NAME).download_file(s3_object.key, new_dir + '/' + filename)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
