# increase_cases_covid19_id_prediction
predict an increase in Covid-19 cases per day in Indonesia

pandemi telah melanda negeri kita tercinta indonesia hampir setengah tahun atau hampir enam bulan dengan dampak yang sangat signifikan bagi kehidupan masyarakat indonesia. hampir semua kegiatan sehari-hari kita berubah setelah pandemi ini melanda indonesia dan melanda seluruh belahan dunia.

tak terkecuali bagi mahasiswa ataupun pelajar, dimasa pandemi ini mahasiswa dan pelajar menempuh kegiatan pendidikannya lewat daring atau dalam jaringan. bisa dikatakan menempuh pendidikan dengan daring yang terjadi saat ini sangatlah tidak efisien karena kualitas internet yang kurang baik dan kurang meleknnya mahasiswa atau pelajar terhadap teknologi informasi saat ini.

bagi mahasiswa pasti sudah terbiasa dengan pengoprasian komputer, bagaimana dengan para pelajar, apalagi pelajar smp ataupun sma, sebagian atau sedikit dari mereka belum terbiasa dengan pengoprasian komputer. terpaksa mereka harus menggunakan smartphone mereka supaya dapat mengikuti pembelajaran secara daring.

tetapi dalam pandemi yang sedang kita alami ini jangan membuat kita pantang menyerah dan jangan sampai kita merasa menyesal dengan adanya pandemi ini. semua kejadian baik atau buruk pasti ada hikmah nya, hikmahnya kita ambil dan jadikan pelajaran bagi kita.

teknologi yang semakin maju, semakin banyak lagi hal-hal yang dapat kita pelajari di dunia ini, banyak sekali informasi-informasi penting atau pembelajaran yang sangat bermanfaat bagi kita contohnya adalah pembelajaran machine learning.

machine learning merupakan pembelajaran algoritma komputer yang dapat dapat bekerja dengan sendirinya. contohnya memprediksi data dan mengklasifikasi data.
kali ini kita akan mencoba memprediksi kenaikan kasus harian covid-19 indonesia meliputi seluruh provinsi di indonesia menggunakan penerapan machine learning berbasis bahasa pemograman python. 

data yang kita dapat yaitu data covid-19 indonesia dari https://www.kaggle.com/hendratno/covid19-indonesia
data tersebut berbasis csv, csv atau comma separated value merupakan sebuah format file yang berisi kumpulan data dengan pemisah koma.
pertama-tama kita memanggil pustaka python yang kita butuhkan
```
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
import pandas as pd
```
lalu kita membuat variabel dataframe bernama df dengan membaca file csv yang kita dapat dari kaggle
```
df = pd.read_csv("covid_19_indonesia_time_series_all.csv", delimiter=",")
```
kemudian kita memfokuskan kolom yang akan kita eksekusi yaitu kolom bernama NewCase yaitu kolom yang berisi kenaikan covid-19 perhari, kita buat variabel bernama case_per_day , variabel ini akan berisi kolom tanggal dan rata-rata kasus baru perhari, kita merata-ratakan kasus harian covid 19 dari semua provinsi.
```
case_per_day = df.groupby("Date").NewCases.mean().reset_index()
#print(case_per_day)
```
selanjutnya kita akan memisahkan kedua kolom tersebut ke variabel Xdan variabel y , untuk X berisi kolom tanggal dan untuk y berisi kolom kasus covid-19 harian
```
X = case_per_day["Date"]
X = X.values.reshape(-1, 1)
y = case_per_day["NewCases"]
```
kita telah membagi dua kolom menjadi 2 variabel, lalu kita buat plot dengan menggunakan scatter plot
```
plt.scatter(X, y)
```
sekarang kita membangun model kita yaitu model linear regression, kita buat variabel bernama regression , setelah membuat model regression lalu membuat metode .fit() , metode ini berisi variabel X dan y
```
regression = LinearRegression()
regression.fit(X,y)
```
kita ingin melihat koefisien dan intercept dari model yang telah dibuat
```
print(regression.coef_)
print(regression.intercept_)
```
output:
[0.39376308]
52.8153723940169
selanjutnya kita akan melihat garis linear regression dengan cara membuat variabel y_predict yang berisi metode .predict()
```
y_predict = regression.predict(X)
```
selanjutnya menampilkan garis linear regression dan mengatur titik-titik data untuk tanggal dan memberi label pada x dan y lalu menampilkannya dengan metode .show()
```
ax = plt.subplot()

plt.plot(X, y_predict)
ax.set_xticks([1,5,10,15,20,25])
ax.set_xticklabels(["1","5","10","15","20","25"])

plt.xlabel("tanggal di bulan agustus")
plt.ylabel("kasus perhari")

plt.show()
```
pada grafik tersebut terlihat bahwa garis linear sedikit menaik yang berarti dari hari ke hari rata-rata kasus covid-19 indonesia mengalami kenaikan.
mari kita prediksi pertambahan kasus covid-19 beberapa hari kedepan, kita buat variabel bernama X_future
```
X_future = np.array(range(26, 40))
X_future = X_future.reshape(-1, 1)
```
pada visualisasi data kita tambahkan plot prediksi
```
plt.plot(X_future, future_predict)
```
kita tambahkan tanggal pada labels lalu tampilkan dengan .show()
```
ax.set_xticks([1,5,10,15,20,25,30,35,40])
ax.set_xticklabels(["1","5","10","15","20","25","30","5","10"])

plt.xlabel("tanggal di bulan agustus dan september")
plt.ylabel("kasus perhari")
plt.show()
```
garis prediksi yang berwarna orange menunjukan bahwa beberapa hari kedepan rata-rata kasus harian covid-19 indonesia akan terus naik, selama pertambahan naik terus, kita harus tetap waspada, jaga kesehatan, jaga jarak dan selalu makan makanan bergizi karena kita tahu bahaya pasti akan datang kapan saja.
semoga pandemi ini segera berakhir supaya kita semua kembali ke kondisi normal seperti sediakala dengan kondisi badan yang sehat dan fit untuk menjalani kehidupan yang bahagia.
