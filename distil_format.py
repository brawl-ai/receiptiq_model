import json

with open("batch_689b402f9e24819088ace24f4247c6ad_output.jsonl", "r", encoding="utf-8") as f:
    content = f.read()
    for response in content.splitlines():
        response_json = json.loads(response)
        image_file = response_json["custom_id"]
        image_name = image_file.replace("datasets/images2/","")
        extracted_content = response_json["response"]["body"]["choices"][0]["message"]["content"]
        try:
            data = json.loads(extracted_content)
            with open(f"datasets/data2/{image_name}.json","w") as f:
                f.write(json.dumps(data))
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON for {image_file}: {e}")
            with open(f"datasets/data2/{image_name}.json","w") as f:
                f.write(extracted_content)