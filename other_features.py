import code

train = code.pdf.get_train_file()
test = code.pdf.get_test_file()

def extract_date(df):
    df['created year'] = 0
    df['created month'] = 0
    df['created year'] = df['created'].apply(lambda x: int(str(x)[0:4]))
    df['created month'] = df['created'].apply(lambda x: int(str(x)[5:7]))
    return df

train = extract_date(train)

print(train)