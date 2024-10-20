import requests
import json
import gradio as gr

url="http://localhost:11434/api/generate"

headers={
    'Content-Type':'application/json'
}

history=[]

def generate_response(prompt):

    history.append(prompt)
    final_prompt="\n".join(history)

    data={
        "model":"codeguru",
        "prompt":final_prompt,
        "stream":False
    }

    response=requests.post(url,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response']
        return actual_response
    else:
        print("error:",response.text)


'''
The API at https://api.example.com/generate will receive this data 
and use it to determine what model to use ("codeguru"), 
what input to process (final_prompt), 
and whether to stream the output (False).

'''


interface=gr.Interface(

    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your prompt"),
    outputs="text"


)

interface.launch()