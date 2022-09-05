import os
import heapq

class HuffmanCoding:

    # during instantiation ask the path of the file that is going to be compressed
    def __init__(self, path):
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_codes = {}

    class HeapNode:

        # each node consists of a character and its frequency 
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            # left and right children of the node
            self.left = None
            self.right = None
        
        # overriding functions __lt__ and __eq__ so that we can make comparisons between the nodes
        def __lt__(self, other):
            if self.freq < other.freq:
                return True
            else:
                return False
        
        def _eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, 'HeapNode')):
                return False
            if self.freq == other.freq:
                return True 
            else:
                return False
        
    def create_freq_dict(self, text):
        """
        Function that creates a dictionary which contains the frequency of each character in the given text.
        """
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            else:
                frequency[character] += 1

        return frequency
    
    def create_heap(self, frequency):
        """
        Function that creates a priority queue containing the characters, based on their frequency.
        """
        for key in frequency:
            # instantiate a HeapNode object for each key in the frequency dictionary
            node = self.HeapNode(key, frequency[key])
            # push the node in the heap
            heapq.heappush(self.heap, node)
    
    def merge_nodes(self):
        """
        Function that actually builds the Huffman tree.
        """
        while (len(self.heap) > 1):
            first_node = heapq.heappop(self.heap)
            second_node = heapq.heappop(self.heap)

            # the merge of the two nodes, created a new node that represents no character and its frequency 
            # is the sum of the frequency of the two parent nodes
            child_node = self.HeapNode(None, first_node.freq + second_node.freq)
            child_node.left = first_node
            child_node.right = second_node

            # insert the node that was created in the heap
            heapq.heappush(self.heap, child_node)
    
    def create_codenames_help(self, node, current_code):
        """
        Recursive function to help us create the codename for each character in the text.
        """
        if (node == None):
            return
        
        if (node.char != None):
            # save the codes in the dictionaries
            self.codes[node.char] = current_code
            self.reverse_codes[current_code] = node.char
        
        self.create_codenames_help(node.left, current_code + "0")
        self.create_codenames_help(node.right, current_code + "1")

    def create_codenames(self):
        """
        Function that after the Huffman tree has been built, creates a code for each character.
        """
        root = heapq.heappop(self.heap)
        current_code = ""
        self.create_codenames_help(root, current_code)

    def get_encoded_text(self, text):
        """
        Function that replaces the characters in the text with their corresponing codenames.
        """
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]

        return encoded_text
    
    def pad_encoded_text(self, encoded_text):
        """
        Pad the encoded text with 0's until its length is a multiple of 8.
        """
        # find how many numbers have to be added 
        extra_padding = 8 - len(encoded_text) % 8

        for i in range(extra_padding):
            encoded_text += "0"

        # store how many numbers were added so that they can later be removed by the decoder
        info = "{0:08b}".format(extra_padding)
        encoded_text = info + encoded_text

        return encoded_text
    
    def get_byte_array(self, padded_encoded_text):
        """
        Convert bits into bytes and return the byte array.
        """
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b
    
    def encode(self):
        """
        Encoding function - encodes the given file and saves it in a .bin file using the functions that were created above.
        """
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + ".bin"

        with open(self.path, 'r') as file, open(output_path, 'wb') as output:
            text = file.read()
            # remove whitespaces from the text given
            text = text.rstrip()

            frequency = self.create_freq_dict(text)
            self.create_heap(frequency)
            self.merge_nodes()
            self.create_codenames()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))
        
        print("Encoded!")
        return output_path

    # FUNCTIONS USED FOR THE DECODING PROCCESS
    
    def remove_padding(self, bit_string):
        """
        Function to remove the padding information that came with the encoded text.
        """
        padded_info = bit_string[:8]
        extra_padding = int(padded_info, 2)

        bit_string = bit_string[8:]
        encoded_text = bit_string[:-1 * extra_padding]

        return encoded_text
    
    def decode_text(self, encoded_text):
        """
        Function used to decode the given encoded text and return it.
        """
        current_code = ""
        decoded_text = ""

        # read the encoded text until a valid codename is found 
        for bit in encoded_text:
            current_code += bit
            # when a valid codename is found, find its corresponding character from the dictionary and replace it with it
            if (current_code in self.reverse_codes):
                character = self.reverse_codes[current_code]
                decoded_text += character
                current_code = ""
        
        return decoded_text
    
    def decode(self, input_path):
        """
        Function that decodes the file given in the input_path using the above functions.
        """
        filename, file_extension = os.path.splitext(input_path)
        output_path = filename + "_decoded" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)
            decoded_text = self.decode_text(encoded_text)

            output.write(decoded_text)

        print("Decoded!")
        return output_path
