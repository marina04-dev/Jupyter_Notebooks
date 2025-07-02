import pandas as pd
import streamlit as st
import altair as alt # Visualization library for creating charts
from PIL import Image # Python Imaging Library for handling images


# Page Title
image  = Image.open('dna-logo.jpg') # Opens an image file from your directory

# Displays the image, automatically sizing it to fit the column width
st.image(image, use_column_width=True)

st.write("""
# DNA Nucleotide Count Web App
         
This app counts the nucleotide composition of query DNA!
         
***
""")
# ***:  creates a horizontal divider line

# Input Text Box
# st.sidebar.header('Enter DNA sequence:')
st.header('Enter DNA Sequence:')

# Default DNA sequence in FASTA format
sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"


#sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
# Creates a multi-line text input box
# "Sequence input" - Label for the text area
# sequence_input - Default text to show
# height=250 - Height of the text box in pixels
sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines() #  Splits the text into a list, one item per line
sequence = sequence[1:] # Takes all lines except the first one (removes FASTA header)
sequence = ''.join(sequence) # Combines all lines into one continuous string with no spaces

st.write("""
***
""")

## Shows the processed DNA sequence (just the nucleotides, no header)
st.header('INPUT (DNA Query)')
sequence # In Streamlit, writing a variable name alone displays its value


## DNA nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

### 1. Print dictionary
st.subheader('1. Print dictionary')
def DNA_nucleotide_count(seq):
  d = dict([
            ('A',seq.count('A')),
            ('T',seq.count('T')),
            ('G',seq.count('G')),
            ('C',seq.count('C'))
            ])
  return d

X = DNA_nucleotide_count(sequence)

#X_label = list(X)
#X_values = list(X.values())

X

### 2. Print text
st.subheader('2. Print text')
st.write('There are  ' + str(X['A']) + ' Adenine (A)')
st.write('There are  ' + str(X['T']) + ' Thymine (T)')
st.write('There are  ' + str(X['G']) + ' Guanine (G)')
st.write('There are  ' + str(X['C']) + ' Cytosine (C)')

### 3. Display DataFrame
st.subheader('3. Display DataFrame')
# Creates DataFrame from dictionary, using keys as row indices
df = pd.DataFrame.from_dict(X, orient='index')
# Renames the first column to 'count'
df = df.rename({0: 'count'}, axis='columns')
# Moves the index (A,T,G,C) into a regular column
df.reset_index(inplace=True)
# Renames the index column to 'nucleotide'
df = df.rename(columns = {'index':'nucleotide'})
# Result: A table with 'nucleotide' and 'count' columns
st.write(df)

### 4. Display Bar Chart using Altair
st.subheader('4. Display Bar chart')
# Creates an Altair chart object using our DataFrame
# .mark_bar() - Specifies this will be a bar chart
# .properties(width=alt.Step(80)) - Sets each bar to be 80 pixels wide
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)
p = p.properties(
    width=alt.Step(80)  # controls width of bar.
)
st.write(p)
