
To init
uv sync 

Put in the good env
.venv 

To test well define 
uv run main.py

What do I do : 


https://medium.com/data-science-collective/zero-shot-forecast-with-moirai-moe-f81c764bf0e2
https://github.com/anamabo/medium-blogs/tree/main/moirai-moe

https://github.com/SalesforceAIResearch/uni2ts
https://github.com/SalesforceAIResearch/uni2ts/tree/main/example
https://huggingface.co/Salesforce/moirai-1.0-R-large

https://github.com/SalesforceAIResearch/uni2ts/tree/main/src/uni2ts/model/moirai
- forecast.py
- module.py

I implement MoiraiEncoder in encoder.py using MoiraiModule and MoiraiForecaster from https://github.com/SalesforceAIResearch/uni2ts/tree/main/src/uni2ts/model/moirai 
- forecast.py
- module.py


In the initial code : 
1. Apply scaling to observations
2. Project from observations to representations
3. Replace prediction window with learnable mask
4. Apply transformer layers
5. Project from representations to distribution parameters
6. Return distribution object

I drop 5. 6., 

Moreover model take as input processed inputs. So i keep the _convert method from MoiraiForecaster.   


TO test : 
moirai.ipynb
