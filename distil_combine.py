import os
import json

def get_schema(data: dict):
    schema = {}
    try:
        for k,v in data.items():
            if isinstance(v, dict):
                if "value" in v:
                    field_type = "string" if isinstance(v["value"], str) else "number"
                    field_description = v["description"] if v["description"] else k
                    schema[k] = f"{field_type}//{field_description}"
                else:
                    schema[k] = get_schema(v)
            elif isinstance(v,list):
                if len(v)>0:
                    schema[k] = [get_schema(v[0])] 
                else:
                    raise Exception(f"found {k} that is an empty list")
            else:
                    raise Exception(f"found {k}:{v} that is has unexpected type {type(v)}")
        return schema
    except Exception as e:
        raise

def remove_descr(data: dict):
    out_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            if "value" in v:
                vc = v.copy()
                vc.pop("description",None)
                out_data[k] = vc
            else:
                out_data[k] = remove_descr(v)
        elif isinstance(v, list):
            out_data[k] = [remove_descr(child) for child in v]
    return out_data


dataset = []
for image in os.listdir("datasets/images2/"):
    json_data_filepath = f"datasets/data2/{image}.json"
    if os.path.exists(json_data_filepath):
        with open(json_data_filepath, "r") as f:
            extracted_data = json.load(f)
            try:
                schema = get_schema(extracted_data)
                data = remove_descr(extracted_data)
                dataset.append({
                    "image_name": image,
                    "schema": schema,
                    "data": data
                })
            except Exception as e:
                print(f"File {json_data_filepath} schema has issue: {e}")
                break

with open("datasets/receipts_dataset2.json", "w") as f:
    f.write(json.dumps(dataset))

print(len(dataset))
print()
print(dataset[0])
print()
print(dataset[0]["data"])
print()
print(dataset[0]["schema"])
