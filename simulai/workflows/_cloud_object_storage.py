# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os.path

from minio import Minio
from minio.error import S3Error


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


class CloudObjectStorage:
    def __init__(self, endpoint, access_key, secret_key):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

    @classmethod
    def from_config_json(cls, cos_config):
        endpoint = _remove_prefix(cos_config["url"], "https://")

        return cls(
            endpoint=endpoint,
            access_key=cos_config["accessKey"],
            secret_key=cos_config["secretKey"],
        )

    def put_file(self, bucket_name, path_in_bucket, local_path):
        while path_in_bucket.startswith("/"):
            path_in_bucket = _remove_prefix(path_in_bucket, "/")

        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
        )
        try:
            client.fput_object(
                bucket_name=bucket_name,
                object_name=path_in_bucket,
                file_path=local_path,
            )
        except S3Error as exc:
            print("Cloud Object Storage error occurred.", exc)
            raise exc

    def get_file(self, bucket_name, path_in_bucket, local_path):
        while path_in_bucket.startswith("/"):
            path_in_bucket = _remove_prefix(path_in_bucket, "/")

        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
        )
        try:
            client.fget_object(
                bucket_name=bucket_name,
                object_name=path_in_bucket,
                file_path=local_path,
            )

        except S3Error as exc:
            print("Cloud Object Storage error occurred.", exc)
            raise exc

    def check_file_exists(self, bucket_name, path_in_bucket):
        while path_in_bucket.startswith("/"):
            path_in_bucket = _remove_prefix(path_in_bucket, "/")

        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
        )
        try:
            object_generator = client.list_objects(
                bucket_name=bucket_name, prefix=path_in_bucket
            )

        except S3Error as exc:
            print("Cloud Object Storage error occurred.", exc)
            raise exc

        return path_in_bucket in set([o.object_name for o in object_generator])
