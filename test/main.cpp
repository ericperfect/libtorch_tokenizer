//
// Created by Eric on 2021/1/5.
//

#include <torch/script.h> // One-stop header.
#include<torch/csrc/api/include/torch/utils.h>
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
    torch::jit::script::Module bert;
    const char *model = argv[1];
    bert = torch::jit::load(join(model, "traced_bert.pt"));
    bert.eval();
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
        vector<float> input_ids2;
        vector<float> input_mask2;
        vector<float> segment_ids2;
        vector<vector<float>> input_ids_list;
        vector<vector<float>> input_mask_list;
        vector<vector<float>> segment_ids_list;
        const char *truncation_strategy = "only_first";
        tokenizer.encode(textA, textB, input_ids, input_mask, segment_ids, max_seq_length, truncation_strategy);
        printVector(input_ids);
        printVector(input_mask);
        printVector(segment_ids);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::from_blob(input_ids.data(), {1, max_seq_length}).to(torch::kLong));
        inputs.push_back(torch::from_blob(input_mask.data(), {1, max_seq_length}).to(torch::kLong));
        inputs.push_back(torch::from_blob(segment_ids.data(), {1, max_seq_length}).to(torch::kLong));

        at::Tensor output = bert.forward(inputs).toTensor();
        std::cout << output << '\n';
    }

}
