"""The module for the standard Huffman tree compression."""

class Node:
    """
    Node class for the Huffman tree.
    Represents an element in the Huffman tree.
    """
    def __init__(self, left=None, right=None, value=None, frequency=None, parent=None):
        self.left = left
        self.right = right
        self.value = value
        self.frequency = frequency
        self.parent = parent

    def __str__(self):
        return f"{self.value} : {self.frequency}"

    def __repr__(self):
        return f"{self.value} : {self.frequency}"

    # Compare nodes based on frequency
    def __lt__(self, other):
        return self.frequency < other.frequency

    def __le__(self, other):
        return self.frequency <= other.frequency

    def __eq__(self, other):
        return self.frequency == other.frequency

    def __ne__(self, other):
        return self.frequency != other.frequency

    def __gt__(self, other):
        return self.frequency > other.frequency

    def __ge__(self, other):
        return self.frequency >= other.frequency

    # Check if the node is a leaf node
    def is_leaf(self):
        return self.left is None and self.right is None

    # Check if the node is a root node
    def is_root(self):
        return self.parent is None


class tree:
    """Main class for the Huffman tree."""
    def __init__(self, data=None, root=None):
        self.data = data
        self.root = root
        self.codes = {}
        self.encoded_tree = ""

    # Build the Huffman tree
    def build_tree(self):
        # Get the frequency of each character
        frequency = {}
        for char in self.data:
            if char not in frequency:
                frequency[char] = 0
            frequency[char] += 1

        # Create nodes for each character
        nodes = [Node(value=char, frequency=freq) for char, freq in frequency.items()]
       
        # Build the tree
        while len(nodes) > 1:
            # Sort nodes by frequency
            nodes.sort()

            # Get the two nodes with the lowest frequency
            left = nodes.pop(0)
            right = nodes.pop(0)

            # Create a new node with the two nodes as children
            new_node = Node(left=left, right=right, frequency=left.frequency + right.frequency)
            # Set the parent of the two nodes
            left.parent = new_node
            right.parent = new_node
            # Add the new node to the nodes list
            nodes.append(new_node)

        # Set the root of the tree
        self.root = nodes[0]

    # Get codes for each character
    def get_codes(self, node=None, code=""):
        if node is None:
            node = self.root
        if node.is_leaf():
            self.codes[node.value] = code
        else:
            self.get_codes(node.left, code + "0")
            self.get_codes(node.right, code + "1")

    # Encode tree structure
    def encode_tree(self, node=None):
        if node is None:
            node = self.root
            self.encoded_tree = ""
        if node.is_leaf():
            self.encoded_tree += "1" + bin(ord(node.value))[2:].zfill(8)
        else:
            self.encoded_tree += "0"
            self.encode_tree(node.left)
            self.encode_tree(node.right)

    # Encode tree structure signature
    def encode_tree_signature(self):
        return bin(0)[2:].zfill(8)


class BitWriter:
    def __init__(self, file_path, MY_tree):
        self.file = open(file_path, 'wb')  # Open the file in write-binary mode
        self.tree_data = MY_tree  # The Huffman tree
        self.buffer = 0  # Buffer for writing bits
        self.bit_count = 0  # Counter for bits written per byte

    # Write the tree structure to the file
    def write_tree_structure(self):
        self.write_bits(self.tree_data.encoded_tree)

    # Write data to the file
    def write_data(self, data):
        for char in data:
            self.write_bits(self.tree_data.codes[char])

    # Write bits to the file
    def write_bits(self, bits):
        for i in bits:
            if i == "1":
                self.buffer = (self.buffer << 1) | 1
            else:
                self.buffer = (self.buffer << 1) | 0
            self.bit_count += 1
            # If 8 bits accumulated, write them to the file
            if self.bit_count == 8:
                self.file.write(bytes([self.buffer]))
                self.buffer = 0
                self.bit_count = 0

    # Close the file, handling leftover bits
    def close(self):
        # If there are leftover bits, pad them to form a full byte
        if self.bit_count > 0:
            # Pad the remaining bits in the buffer to form a full byte
           # self.buffer <<= (8 - self.bit_count)  # Shift to the left to add trailing zeros
            self.file.write(self.buffer.to_bytes(1, 'big'))  # Write the padded byte

            # Create the map byte, where the number of '1's matches the leftover bits
            map_byte = (1 << self.bit_count) - 1  # e.g., if bit_count = 4 -> map_byte = 0b1111
            map_byte <<= (8 - self.bit_count)  # Shift to align the '1's to the left
            self.file.write(map_byte.to_bytes(1, 'big'))  # Write the map byte
        else:
            # No leftover bits, write a full "valid" map byte
            self.file.write((0b11111111).to_bytes(1, 'big'))  # Indicate all 8 bits are valid

        # Close the file
        self.file.close()


class BitReader:
    def __init__(self, file_path):
        self.file = open(file_path, 'rb')  # Open the file in binary read mode
        self.buffer = 0  # Buffer to hold the current byte being processed
        self.bit_count = 0  # Tracks how many bits have been read from the buffer
        self.remaining_bits = None  # Number of valid bits in the padded byte
        self.file_size = self.file.seek(0, 2)  # Get the size of the file
        self.file.seek(0)  # Reset the file pointer to the beginning
        self.file_ended = False  # Flag to indicate the end of meaningful data

    def read_bit(self):
        if self.file_ended:
            return None

        if self.bit_count == 0:
            byte = self.file.read(1)
            if not byte:  # If we’ve reached the end of the file unexpectedly
                self.file_ended = True
                return None

            # Check if we're at the padding and map bytes
            if self.file.tell() == self.file_size:
                # Handle missing map/padding bytes gracefully
                self.file.seek(-2, 2)  # Move to the last two bytes
                map_byte = self.file.read(1)
                padded_byte = self.file.read(1)

                if not map_byte or not padded_byte:
                    self.file_ended = True
                    return None

                map_byte = ord(map_byte)
                padded_byte = ord(padded_byte)

                self.remaining_bits = bin(map_byte).count('1')
                self.buffer = (padded_byte >> (8 - self.remaining_bits))
                self.bit_count = self.remaining_bits
                self.file_ended = True
                return None

            self.buffer = ord(byte)
            self.bit_count = 8

        self.bit_count -= 1
        return (self.buffer >> self.bit_count) & 1
    

class BitReader:
    def __init__(self, file_path):
        self.file = open(file_path, 'rb')  # Open the file in binary read mode
        self.buffer = 0  # Buffer to hold the current byte being processed
        self.bit_count = 0  # Tracks how many bits have been read from the buffer
        self.file_ended = False  # Flag to indicate the end of the file
        self.file_size = self.file.seek(0, 2)  # Get the size of the file
        self.file.seek(0)  # Reset the file pointer to the start

    def read_bit(self):
        if self.bit_count == 0:
            if self.endfile():  # Check if we're at the padding byte
                byte = self.file.read(1)  # Read the last data byte
                map_byte = self.file.read(1)  # Read the mapping byte 
                if not map_byte:  # If map_byte is empty, end of the file reached
                    self.file_ended = True
                    return None

                map_byte_int = ord(map_byte)
                self.bit_count = sum(1 for bit in bin(map_byte_int)[2:] if bit == '1')

            else:  # Read a normal byte
                byte = self.file.read(1)
                self.bit_count = 8

            if not byte:  # If no more bytes to read, end the file
                self.file_ended = True
                return None

            self.buffer = ord(byte)

        self.bit_count -= 1
        return (self.buffer >> self.bit_count) & 1

    def read_bits(self, num_bits):
        bits = []
        for _ in range(num_bits):
            bit = self.read_bit()
            if bit is None:
                break
            bits.append(str(bit))
        return ''.join(bits)

    def endfile(self) -> bool:
        """
        Check if the current position is at the padding and mapping bytes.
        Returns True if the file pointer is positioned before the last two bytes.
        """
        current_pos = self.file.tell()
        # Check if we're within the last two bytes of the file
        return current_pos >= self.file_size - 2

    def close(self):
        self.file.close()


    def read_bits(self, num_bits):
        # Read multiple bits at a time
        bits = []
        for _ in range(num_bits):
            bit = self.read_bit()
            if bit is None:
                break
            bits.append(str(bit))
        return ''.join(bits)

    def close(self):
        # Close the file
        self.file.close()


def decode_tree(bit_reader):
    bit = bit_reader.read_bit()
    if bit == 1:  # Leaf node
        char_bits = bit_reader.read_bits(8)
        if char_bits == 0 :  # NULL case stop; the tree is done
            return None
        char = chr(int(char_bits, 2))
        return Node(value=char)
    else:  # Internal node
        left_child = decode_tree(bit_reader)
        right_child = decode_tree(bit_reader)
        return Node(left=left_child, right=right_child)


def decode_data(bit_reader, root):
    decoded_text = []
    current_node = root
    while not bit_reader.file_ended:
        bit = bit_reader.read_bit()
        if bit is None:
            break
        current_node = current_node.left if bit == 0 else current_node.right
        if current_node.is_leaf():
            decoded_text.append(current_node.value)
            current_node = root
    return ''.join(decoded_text)


def compress(data):
    MY_tree = tree(data=data)
    MY_tree.build_tree()
    MY_tree.get_codes()
    MY_tree.encode_tree()
    MY_tree.encode_tree_signature()

    writer = BitWriter("compressed.bin", MY_tree)
    writer.write_tree_structure()
    writer.write_data(data)
    writer.close()


def decompress(file_path):
    bit_reader = BitReader(file_path)
    root = decode_tree(bit_reader)
    decoded_text = decode_data(bit_reader, root)
    bit_reader.close()
    return decoded_text


# Main functionality
if __name__ == "__main__":
    with open('inputhuffman.txt', "r") as f:
        data = f.read()
        compress(data)
    with open("outputhuffman.txt", 'w') as f:
        decompressed_data = decompress('compressed.bin')
        f.write(decompressed_data)

