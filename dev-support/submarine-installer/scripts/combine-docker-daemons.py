import json
import sys 

def combineJsons(jsonFile1, jsonFile2, outputFile): 
  dict1 = json.load(open(jsonFile1)) 
  dict2 = json.load(open(jsonFile2))
  dict3 = dict(dict1.items() + dict2.items())
   
  with open(outputFile, 'w') as output:
    json.dump(dict3, output, indent=2, sort_keys=True)
  
  return True
  
if __name__ == '__main__':
  if (len(sys.argv) < 4):
    raise Exception,u"3 arguments needed"
  
  print(combineJsons(sys.argv[1], sys.argv[2], sys.argv[3]))
