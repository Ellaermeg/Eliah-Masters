from bioservices import KEGG
kegg = KEGG()

def translate_ko_terms(ko_terms):
    ko_names = {}
    for ko in ko_terms:
        try:
            info = kegg.get(ko)  # Retrieve information for the KO term
            definition = kegg.parse(info)['DEFINITION']  # Parse the information to get the definition
            ko_names[ko] = definition.split(';')[0]  # Sometimes definitions contain multiple parts; take the first
        except Exception as e:
            print(f"Error retrieving information for {ko}: {e}")
            ko_names[ko] = ko  # Use the KO term itself if the name can't be retrieved
    return ko_names