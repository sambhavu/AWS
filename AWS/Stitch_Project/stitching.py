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


    def download_image(self, path, file_name):
        client = self.connectAWS_client()
        try:
            client.download_file(self.Bucket, path, file_name)
            print("Download Successful: \nAWS Path: ",path, "\nFile Name: ", file_name)

        except Exception as e:
            print("Error Occurred: Download Failed path: ", path)
            print(e)


    def get_picture_path(self, file_name):
        client = self.connectAWS_client()
        file_number = file_name[-4]
        print(file_number)

        objects = client.list_objects(Bucket = self.Bucket, Prefix = PATH_JPEG)

        for content in objects.get('Contents', []):
            raw = content.get('Key')


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

        return file[pos_f:pos_l + 1 ], file[pos_f:]




    def combine(self, img1, img2, filename):


        image1  = cv2.imread(img1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img2, cv2.IMREAD_COLOR)

        try:
            combined_image = np.concatenate((image1, image2), axis = 1)
        except:
            h, w, c = image1.shape
            image2 = cv2.resize(image2, (w, h))
            combined_image = np.concatenate((image1, image2), axis = 1)

        cv2.imwrite(filename, combined_image)




    def download_image(self, PATH_RAW, PATH_JPEG):
        downloaded_images_count = 0
        errors = 0

        client = self.connectAWS_client()
        objects = client.list_objects(Bucket = self.Bucket, Prefix = PATH_RAW)

     #   numbers = len(objects.get('Contents', []))
     #   print(numbers)

        for content in objects.get('Contents', []):

            raw = content.get('Key')

            print("")
            print(raw,"\nStatus: ", end="")


            if raw.count("-") > 7 and raw.count("-") < 10:

                file_, save_name = self.get_file_name(raw)

                print("Finding Corresponding Image...")

                #Try to Download as PNG, png, JPEG, jpeg

                img1 = '1' + file_
                img2 = '1' + save_name

                try:
                    file_name = file_ + ".PNG"
                    client.download_file(self.Bucket, PATH_JPEG + file_name, img2)

                except:

                    try:

                        file_name = file_ + ".png"
                        client.download_file(self.Bucket, PATH_JPEG + file_name, img2)

                    except:

                        try:

                            file_name = file_ + ".jpg"
                            client.download_file(self.Bucket, PATH_JPEG + file_name, img2)

                        except:

                            try:

                                file_name = file_ + ".jpeg"
                                client.download_file(self.Bucket, PATH_JPEG + file_name, img2)


                            except Exception as e:

                                print("Individual Image Download Failed\nRaw Key: ", raw, "\nJPEG Key: ", PATH_JPEG)
                                print("Exception/Error Occurred: ", e)
                                errors =  errors + 1
                                continue








                print("Corresponding Image Found in /jpg as: ", file_name)


                try:

                    client.download_file(self.Bucket, raw, img1)

                except Exception as e:

                    print("Individual Image Download Failed\nRaw Key: ", raw, "\nJPEG Key: ", PATH_JPEG)
                    print("Exception/Error Occurred: ", e)
                    errors = errors + 1
                    continue



                print("Individual Image Download Successful")



                self.combine(img2, img1, save_name)
                print("Image Saved. File Name: ", save_name)

                os.remove(img1)
                os.remove(img2)
                print("Individual Files Removed")
                downloaded_images_count = downloaded_images_count + 1

            else:
                print("No Image for Key")


        print("Downloaded: ", downloaded_images_count)
        print("Errors: ", errors)












BUCKET = 'bes.covid19.incoming.xrays'
PATH_RAW = 'predicted_images/musc/'
PATH_JPEG = 'jpg/'
os.chdir('/users/satish/desktop/B/')



def main():


    s3 = AWS(BUCKET)
    s3.download_image(PATH_RAW, PATH_JPEG)

main()





