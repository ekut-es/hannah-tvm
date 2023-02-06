/*
 * Copyright (c) 2023 hannah-tvm contributors.
 *
 * This file is part of hannah-tvm.
 * See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ac_int.h>

#include <iostream>

#define SIZE 65

int main(int argc, char const *argv[])
{
    ac_int<SIZE,true> val;


    std::cout << "Size of " << SIZE << " bit ac_int: " << sizeof(val) << std::endl;

    return 0;
}
