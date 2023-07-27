import boto3
import uuid
import os
import numpy as np
import cv2
from matplotlib  import pyplot as plt



class AWS:

    def __init__(self, Bucket):
        self.Bucket = Bucket


    def connectAWS_client(self):
        return boto3.client(service_name='s3',
                            aws_access_key_id='AKIASGQZGIWQX5OL5ZWH',
                            aws_secret_access_key='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',
                            region_name='us-east-2')


    def connectAWS_resource(self):
       return boto3.resource(service_name='s3',
                        aws_access_key_id='AKIASGQZGIWQX5OL5ZWH',
                        aws_secret_access_key='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',
                        region_name='us-east-2')



    def get_file_name(self, file):
        dash        ='-'
        dash_count  = 0
        pos_f       = 0                #first element of number
        pos_l       = 0                #last element of number

        for i in range(0, len(file)):

            if file[i] == dash:
                dash_count = dash_count + 1

            if dash_count == 5:
                pos_f = i + 1
                break

        pos_l = pos_f

        while True:
            if file[pos_l] == dash and file[pos_l + 1] == 'c':
                pos_l = pos_l - 1
                break

            else:
                pos_l = pos_l + 1

        return file[pos_f:pos_l + 1 ] + ".PNG", file[pos_f:]




    def combine(self, img1, img2, filename):
        image1  = cv2.imread(img1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img2, cv2.IMREAD_COLOR)
        combined_image = np.concatenate((image1, image2), axis = 1)
        cv2.imwrite(filename, combined_image)



    def download_image(self, client, PATH_RAW, PATH_JPEG):
        objects = client.list_objects(Bucket = BUCKET, Prefix = PATH_RAW)

        for content in objects.get('Contents', []):
            raw = content.get('Key')

            if raw.count("-") > 7 and raw.count("-") < 10:

                file_name, save_name = self.get_file_name(raw)

                img1 = '1' +  file_name
                img2 = '1' + save_name

                print(file_name)

                try:
                    client.download_file(self.BUCKET, raw, img1)
                    client.download_file(self.BUCKET, PATH_JPEG + file_name, img2 )

                except:
                    file_name = file_name.lower()

                try:
                    client.download_file(BUCKET, raw, img1)
                    client.download_file(BUCKET, PATH_JPEG + file_name, img2 )

                except:

                    pass


                self.combine(img2, img1, save_name)

                os.remove(img1)
                os.remove(img2)

                print("done")
                break



BUCKET = 'bes.covid19.incoming.xrays'
PATH_RAW = 'predicted_images/musc/'
PATH_JPEG = 'jpg/'
os.chdir('/users/satish/desktop/B/')

def main():



    s3 = AWS(BUCKET)
    client = s3.connectAWS_client()
    resource = s3.connectAWS_resource()
    
    my_bucket = resource.Bucket(BUCKET)
    s3.download_image(client, PATH_RAW, PATH_JPEG)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()





#def upload_aws(self, path, file_name):
#    resp_upload = self.s3.upload_file(file_name, self.bucket, path+"/"+file_name)
#    file_url = self.s3.generate_presigned_url('get_object',
#    Params={'Bucket': self.bucket,
#   'Key': path+"/"+file_name},
#    ExpiresIn=604800)
#    os.remove(file_name)
#    return file_url



#def download_image_aws_arr(self, key):
#    file_byte_string = self.s3.get_object(Bucket=self.bucket, Key=key)['Body'].read()
#    return np.fromstring(file_byte_string, np.uint8)



#s3.download_file('bes.covid19.incoming.xrays', 'predicted_images', 'fa99b8ef-e26a-4b2d-bd4e-899260368b01-99-covid19-prob-0.539.png')

#for key in s3.list_objects(Bucket = 'bes.covid19.incoming.xrays')['Contents']:
#    name = key['Key']
#    print(name)


