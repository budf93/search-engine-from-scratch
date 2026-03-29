import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class EliasGammaPostings:
    """
    Class untuk melakukan compression dengan Elias Gamma Encoding.
    Elias Gamma cocok untuk angka kecil dan sangat efisien untuk gap encoding.
    
    Algoritma Elias Gamma untuk angka n:
    1. Tentukan N = floor(log2(n)). Ini adalah jumlah bit setelah bit pertama.
    2. Tuliskan N buah bit '0' sebagai prefix.
    3. Tuliskan n dalam representasi biner (yang pasti panjangnya N+1 bit).
    
    Contoh:
    - n = 1: bin(1)='1', N=0. Code: '1'
    - n = 2: bin(2)='10', N=1. Code: '0' + '10' = '010'
    - n = 5: bin(5)='101', N=2. Code: '00' + '101' = '00101'
    """

    @staticmethod
    def _to_bits(n):
        """
        Mengonversi satu angka n menjadi string bit Elias Gamma.
        """
        if n <= 0:
            # Elias Gamma hanya untuk bilangan bulat positif (n > 0)
            raise ValueError("Elias Gamma hanya mendukung integer positif (> 0)")
        
        # bin(n) menghasilkan string '0b...' maka kita ambil dari index 2
        binary_n = bin(n)[2:] 
        
        # N adalah jumlah bit "sisa" setelah bit paling signifikan (leading 1)
        N = len(binary_n) - 1
        
        # Format Elias Gamma: N buah nol + seluruh bit biner n
        return '0' * N + binary_n

    @staticmethod
    def bit_encode(numbers):
        """
        Mengonversi list of numbers menjadi bytes menggunakan Elias Gamma.
        Karena Elias Gamma berbasis bit, kita perlu mem-pack bit-bit tersebut ke dalam byte.
        """
        # 1. Gabungkan semua bit dari setiap angka
        bit_string = "".join(EliasGammaPostings._to_bits(n) for n in numbers)
        
        # 2. Hitung berapa padding yang dibutuhkan agar panjang bit_string kelipatan 8
        padding_len = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding_len
        
        # 3. Masukkan ke dalam bytearray. 
        # Byte pertama kita gunakan untuk menyimpan padding_len agar decode tahu kapan harus berhenti.
        result = bytearray()
        result.append(padding_len)
        
        # 4. Ambil setiap 8 bit dan ubah menjadi integer (1 byte)
        for i in range(0, len(bit_string), 8):
            byte_val = int(bit_string[i:i+8], 2)
            result.append(byte_val)
            
        return bytes(result)

    @staticmethod
    def bit_decode(encoded_bytes):
        """
        Mengonversi bytes kembali menjadi list of numbers.
        """
        if not encoded_bytes:
            return []
            
        # 1. Ambil padding_len dari byte pertama
        padding_len = encoded_bytes[0]
        # 2. Ambil sisa data bytes-nya
        data = encoded_bytes[1:]
        
        # 3. Ubah setiap byte kembali menjadi string bit 8-karakter
        bit_string = ""
        for b in data:
            # zfill(8) memastikan byte seperti 5 menjadi '00000101'
            bit_string += bin(b)[2:].zfill(8)
            
        # 4. Buang padding nol di akhir bit_string
        if padding_len > 0:
            bit_string = bit_string[:-padding_len]
            
        # 5. Decode bit_string menjadi angka-angka
        numbers = []
        i = 0
        n_bits = len(bit_string)
        while i < n_bits:
            # a. Hitung leading zeros (N)
            N = 0
            while i < n_bits and bit_string[i] == '0':
                N += 1
                i += 1
            
            # Jika sisa string hanya nol (padding yang tidak terbuang atau error), stop
            if i >= n_bits:
                break
            
            # b. Elias Gamma: bit '1' tadi adalah bagian dari angka yang panjangnya N+1 bit
            # Ambil bit dari posisi i sebanyak N+1 bit
            val_str = bit_string[i : i + N + 1]
            numbers.append(int(val_str, 2))
            
            # c. Pindahkan pointer i sebanyak bit yang baru dibaca
            i += N + 1
            
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menggunakan Elias Gamma dan Gap Encoding.
        """
        if not postings_list:
            return b""
        
        # Ubah menjadi gap-based (selisih antar docID)
        # Shift the first docID by +1 because docID can be 0, and Elias Gamma doesn't support 0
        gaps = [postings_list[0] + 1]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1])
            
        return EliasGammaPostings.bit_encode(gaps)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode stream of bytes menjadi postings_list (DocIDs asli).
        """
        if not encoded_postings_list:
            return []
            
        # Decode bytes menjadi gaps
        gaps = EliasGammaPostings.bit_decode(encoded_postings_list)
        
        # Kembalikan gaps menjadi DocIDs asli (prefix sum)
        # Unshift the first docID by -1
        postings = [gaps[0] - 1]
        for i in range(1, len(gaps)):
            postings.append(postings[i-1] + gaps[i])
            
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term frequencies menggunakan Elias Gamma.
        TF tidak menggunakan gap encoding karena nilainya biasanya sudah kecil.
        """
        return EliasGammaPostings.bit_encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode term frequencies dari bytes.
        """
        return EliasGammaPostings.bit_decode(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(f"--- Testing {Postings.__name__} ---")
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list     : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        
        assert decoded_posting_list == postings_list, f"{Postings.__name__}: hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, f"{Postings.__name__}: hasil decoding tidak sama dengan TF list original"
        print("Assertion Passed!\n")
