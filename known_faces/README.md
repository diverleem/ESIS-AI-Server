# Direktori Profil Wajah (Known Faces)

Folder ini digunakan untuk menyimpan foto wajah mahasiswa/orang yang ingin dikenali oleh sistem Face Recognition ESIS.

### Cara Menambahkan Wajah Baru:

1. **Format Penamaan File:**
   - Gunakan format: `NIM_Nama_Lengkap.jpg` atau `Nama_Lengkap.jpg`
   - Contoh:
     - `11230910000004_Agung_Susilo.jpg`
     - `11230910000005_Ahmad_Fauzan.jpg`
     - `11230910000079_Ahmad_Faza.jpg`

2. **Menggunakan Script Pendaftaran Otomatis:**
   - Dari file gambar:
     ```bash
     python enroll_face.py --image "path/to/foto.jpg" --name "Agung Susilo" --nim "11230910000004"
     ```
   - Ambil langsung dari streaming Lab Riset Lantai 5:
     ```bash
     python enroll_face.py --from-stream --name "Agung Susilo" --nim "11230910000004"
     ```

3. **Reload Tanpa Restart Server:**
   Saat program utama `main.py` sedang berjalan, tekan tombol **`R`** pada keyboard untuk memuat ulang profil wajah tanpa perlu restart server.
