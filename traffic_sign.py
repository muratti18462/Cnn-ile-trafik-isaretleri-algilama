import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

os.chdir('/Users/muratdemiralay/Downloads/archive')
#burada çalıştığım klasörü değiştiriyorum
data = []
labels = []
classes = 43
#43 tane trafik işaretim daha sonra dolduracağım data ve label listeleri oluşturuyorum
cur_path = os.getcwd()
#burada join işlemi yapacağım curr path i alıyorum

for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '//'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
#hocam burada yukarda tanımladığım kısımdaki path i oluşturuyorum,daha resimlerimi 30 a 30 luk 
#hallerine çeviriyorum zaten yoksa modelimi çok zorlamış olurum,yani cnn modelime girmeye uygun hale
#getiriyorum,train datamın içindeki bütün resimlere bunu yapıyorum           

data = np.array(data)
labels = np.array(labels)
#os.mkdir('training')

np.save('./training/data',data)
np.save('./training/target',labels)

data=np.load('./training/data.npy')
labels=np.load('./training/target.npy')

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
#datamı train ve test kısımlarına ayırıyorum

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#bu kısımda benim yapay zeka modelime giren ve çıkan herşeyin sayı olması lazım,ama labellar sağa dönüş
#tabelesı 30 km hız tabelası gibi sözcüklerden oluşuyor bu sözcükleri one hot arraylere çeviriyorum 
#

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#bu kısımda conv2d kısmında 2 tane layer katman belirliyorum bu katmanlarda 32 nöron var ,bunlarda
#aktivasyon fonksiyonu kullanıyorum sonra max pooling kısmında conv2d de oluşturduğum parametreleri azaltıyorum
#yine dropput layerın ana amacı overfitting'i engellemek oluyor.Flatten yaptığım kısım aslında oluşturduğum 
#image arraylerini düz bir çizgi haline getiriyorum ve bunları benim klasik yapay zeka ağıma sokuyorum.
#ondan sonra 

epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

#epoch u 20 olarak ayarladım,train datamı ve labelımı modelime eğitmeye başlıyorum.eğtimde asıl amaç
#loss fonksiyonunu düşürmek oluyor ve doğruluğu arttırmak oluyor,ama doğruluğunn hiçbir zaman 
#1 olmasını istemiyorum bu durumda overfitting denilen kavram oluşuyor ezberliyor train datamda çok güzel 
#sonuçlar veriyor ama test datamda yeni resimler gördüğü için iyi sonuç veremiyor
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
#

def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label

X_test, label = testing('Test.csv')
#modelime test datasını sokuyorum
Y_pred = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(label, Y_pred))
#modeli doğruluk scoreuna bakıyorum

model.save("/Users/muratdemiralay/Downloads/archive/training/TSR.h5")
model = load_model('/Users/muratdemiralay/Downloads/archive/training/TSR.h5')


classes = { 0:'Hız limiti (20km/h)',
            1:'Hız limiti (30km/h)', 
            2:'Hız limiti (50km/h)', 
            3:'Hız limiti (60km/h)', 
            4:'Hız limiti (70km/h)', 
            5:'Hız limiti (80km/h)', 
            6:'Hız limitinin bittiği yer(80km/h)', 
            7:'Hız limiti(100km/h)', 
            8:'Hız limiti(120km/h)', 
            9:'Girilmez', 
            10:'3.5 tonun üzerindeki araçlar giremez', 
            11:'Right-of-way at intersection', 
            12:'Anayol', 
            13:'Yol ver', 
            14:'Dur', 
            15:'Araç giremez', 
            16:'3.5 tonun üzerindeki araçlar giremez', 
            17:'Girilmez', 
            18:'Dikkat edin', 
            19:'Sola tehlikeli viraj', 
            20:'Sağa tehlikeli viraj', 
            21:'Zikzaklı yol', 
            22:'Kasisli yol', 
            23:'Kaygan yol', 
            24:'Sağdan daralan kaplama', 
            25:'Dikkat yol çalışması var', 
            26:'Trafik ışıkları', 
            27:'Yaya geçidi', 
            28:'Okul geçidi', 
            29:'Bisiklet geçebilir', 
            30:'Gizli buzlanma',
            31:'Yabani hayvan geçebilir', 
            32:'Bütün yasak ve kısıtlamaların sonu', 
            33:'İlerden sağa mecburi dönüş', 
            34:'İlerden sola mecburi dönüş', 
            35:'İleri mecburi yön', 
            36:'İleri veya sağa mecburi yön', 
            37:'İleri veya sola mecburi yön', 
            38:'Sağdan gidiniz', 
            39:'Soldan gidiniz', 
            40:'Ada etrafında dönünüz', 
            41:'Geçme yasağı sonu', 
            42:'3.5 tonun üzerindeki araçlar için geçme yasağı sonu' }

#Burada datasetimizin içindeki trafik işaretlerini bir dictionary içine atıyorum,datasetim için 43 tane sınıf var


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return image,Y_pred

#yukarıda test datamızda yaptığım işlemlerin aynısını burada da yapıyorum

plot,prediction = test_on_img(r'/Users/muratdemiralay/Downloads/archive/Test/00393.png')
s = [str(i) for i in prediction] 
a = int("".join(s)) 
print("Tahmin edilen trafik işareti: ", classes[a])
plt.imshow(plot)
plt.show()

