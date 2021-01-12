//
// Created by Eric on 2021/1/5.
//

//#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <bits/stdc++.h>

#include "tokenizer.h"

using namespace std;

BertTokenizer tokenizer;
BasicTokenizer basictokenizer;

int max_seq_length = 20;

string join(const char *a, char *b)
{
    string message;
    message.reserve(strlen(a) + 1 + strlen(b));
    message = a;
    message += "/";
    message += b;
    return message;
}



void tokenize(string text, vector<string> &tokens, vector<float> &valid_positions)
{
    vector<string> words = basictokenizer.tokenize(text);
    vector<string> token;
    vector<string>::iterator itr;
    for (itr = words.begin(); itr < words.end(); itr++)
    {
        token = tokenizer.tokenize(*itr);
        tokens.insert(tokens.end(), token.begin(), token.end());
        for (int i = 0; i < token.size(); i++)
        {
            if (i == 0)
                valid_positions.push_back(1);
            else
                valid_positions.push_back(0);
        }
    }
}

void encode(string text, vector<float> &input_ids, vector<float> &input_mask, vector<float> &segment_ids,
                vector<float> &valid_positions)
{
    vector<string> tokens;
    tokenize(text, tokens, valid_positions);
    // insert "[CLS}"
    tokens.insert(tokens.begin(), "[CLS]");
    valid_positions.insert(valid_positions.begin(), 1.0);
    // insert "[SEP]"
    tokens.push_back("[SEP]");
    valid_positions.push_back(1.0);
    for (int i = 0; i < tokens.size(); i++)
    {
        segment_ids.push_back(0.0);
        input_mask.push_back(1.0);
    }
    input_ids = tokenizer.convert_tokens_to_ids(tokens);
    while (input_ids.size() < max_seq_length)
    {
        input_ids.push_back(0.0);
        input_mask.push_back(0.0);
        segment_ids.push_back(0.0);
        valid_positions.push_back(0.0);
    }
}

template<typename T>
void printVector(vector<T>& v) {

	for (typename vector<T>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}


int main(int argc, const char *argv[])
{
    const char *model = argv[1];
    string text;
    tokenizer.add_vocab(join(model, "vocab.txt").c_str());
    while (true)
    {
        cout << "\n" << "Input -> ";
        getline(cin, text);
        vector<float> input_ids;
        vector<float> input_mask;
        vector<float> segment_ids;
        vector<float> valid_positions;
        vector<string> tokens;
        tokens = tokenizer.tokenize(text);
        encode(text, input_ids, input_mask, segment_ids, valid_positions);
        printVector(tokens);
        printVector(input_ids);
        printVector(input_mask);
        printVector(segment_ids);
        printVector(valid_positions);
    }

}
