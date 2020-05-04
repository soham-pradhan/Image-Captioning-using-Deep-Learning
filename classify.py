#!C:\Users\Ashish\Anaconda3\python.exe
import subprocess
import sys
#print(sys.argv[1])
out = subprocess.Popen(['python.exe','classify_image.py','--image_file',sys.argv[1]], 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
stdout,stderr = out.communicate()
stdout = stdout.decode('utf-8')
o = stdout.split('\n');
o = o[-6:-1]
predictions = []
for pred in o:
    predictions.append(pred.split('(')[0])


predictions = predictions[0].split(',')

pred = predictions[0].lower()
if pred=="granny smith ":
    print("Apple")
elif pred=="tabby":
    print("Cat")
elif pred=="labrador retriever " or pred=="german shepherd":
    print("Dog")
else:
    print(predictions[0].capitalize())