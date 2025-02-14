# RAPTOR-PDF

### A simple Python3 implementation for RAG(Retrieval Augmentation Generation)-based PDF document content search with RAPTOR(Recursive Abstractive Processing for Tree-like Object Retrieval) algorithm

> This repository is for show **how RAPTOR algorithm can be effectively used** for searching contents within a given PDF document. By recursively processing the tree-like structure of the PDF document, RAPTOR algorithm can be used to search for the contents within the document. (Not an elegant or fancy code, I admit that. It's just showing it works. **Have fun, that's all.**)

### How to use

1. Ensure that you have Python3 and installed with the required libraries enlisted in `requirements.txt`. Then, create the environment variable file `.env` at the same directory of this project, and register your OpenAI API key with a name of `OPENAI_API_KEY` like below. 
```
OPENAI_API_KEY=sk-proj-...
```
3. Execute the program like
```
python3 main.py <path-to-pdf-file>
```
4. If everything goes well(Reading PDF file, Splitting data into pieces, Embedding, Processing chunks via RAPTOR algorith, etc.), you will see the interface like below. I used [**Cisco** Cyber Threat Trends Report(2024)](https://www.cisco.com/c/dam/en/us/products/collateral/security/cyber-threat-trends-report.pdf).
```
Enter your question: According to the given document, how many security events do Cisco observe?
Answer: 
--- Retrieved Documents ---
Document 1:
All rights reserved. 14


Cyber Threat Trends Report
Implementing Security Defense Strategy
• Patch and update systems: Keep all systems • Plan incident response: Develop and regularly
and software updated with the latest patches test an incident response plan so that your
to protect against known vulnerabilities that organization is prepared to respond effectively to
...
------------------------------
Document 4:
The provided documentation discusses a classification of highly skilled threat actors who are focused on espionage and intellectual property theft. These actors have the resources, time, and dedication to carry out sophisticated attacks and are able to remain undetected within networks for extended periods of time. This makes them a persistent and continually evolving threat in the field of cybersecurity. The document is copyrighted by Cisco and/or its affiliates.
------------------------------
---------------------------

According to the given document, Cisco observes 550 billion security events every day.

```
### Note
- Since it essentially uses LLM model to build a RAPTOR tree and facilitiate QnA interface. It means it **costs**. (Initially, I set the model as `gpt-3.5-turbo`, so it won't cost too much for a properly-sized PDF document, actually).
- Depending on the size of document, the running time(preparation time) may vary.

### TO-DO (If I have further time)
- Save the RAPTOR tree data(If I can) and reuse for the same document if there is a duplicated request.

