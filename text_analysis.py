from ccg_nlpy import remote_pipeline

pipeline = remote_pipeline.RemotePipeline(server_api='http://austen.cs.illinois.edu:5800/')

doc = pipeline.doc("Hello, how are you. I am doing fine")
print(doc.get_lemma)
print(doc.get_pos)