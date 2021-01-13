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

int max_seq_length = 512;

string join(const char *a, const char *b)
{
    string message;
    message.reserve(strlen(a) + 1 + strlen(b));
    message = a;
    message += "/";
    message += b;
    return message;
}

template<typename T>
void printVector(vector<T> &v)
{

    for (typename vector<T>::iterator it = v.begin(); it != v.end(); it++)
    {
        cout << *it << " ";
    }
    cout << endl;
}


int main(int argc, const char *argv[])
{
    const char *model = argv[1];
    string textA;
    string textB;
    tokenizer.add_vocab(join(model, "vocab.txt").c_str());
    while (true)
    {
        cout << "\n" << "Input A -> ";
        getline(cin, textA);
        cout << "\n" << "Input B -> ";
        getline(cin, textB);
        vector<float> input_ids;
        vector<float> input_mask;
        vector<float> segment_ids;
        vector<string> tokens;
        const char *truncation_strategy = "only_first";
        tokenizer.encode(textA, textB, input_ids, input_mask, segment_ids, max_seq_length, truncation_strategy);
        printVector(tokens);
        printVector(input_ids);
        printVector(input_mask);
        printVector(segment_ids);
    }

}
