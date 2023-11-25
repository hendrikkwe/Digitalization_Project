import json
import pandas as pd

def add_climate_related_label(object: pd.Series) -> pd.Series:
    pass

def add_domain_label(object: pd.Series) -> pd.Series:
    pass

def add_ron_label(object: pd.Series) -> pd.Series:
    pass

def extract(pdf: str) -> pd.DataFrame:
    
    pdf = pd.DataFrame(columns=["pdf_name","text","page_nr", "climate_related", "domain" ,"r.o.n.", "transition"])

    return pdf

# pdf_name | text | page_nr | climate_related | domain | r.o.n. | ?transition 
#                           | no              | nan 

if "__main__" == __name__:

    # extraction: pdf -> json of pages 
    document = extract()

    for object in document: 
        
        # climate related label
        add_climate_related_label(object)

        if climate_related: 

            # domain 

            if domain == strategy: 
                
                # ron 

                if risk:

                elif opp:
