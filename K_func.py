from bioservices import KEGG
kegg = KEGG()

def translate_ko_terms_with_find(ko_terms):
    ko_descriptions = {}
    for ko in ko_terms:
        try:
            result = kegg.find("ko", ko)
            # The result can contain multiple lines if the term has multiple entries; split by newline
            first_line = result.split('\n')[0]
            # Each line is tab-separated with the format: entry_id, description
            _, description = first_line.split('\t', 1)
            ko_descriptions[ko] = description
        except Exception as e:
            print(f"Error retrieving information for {ko}: {e}")
            ko_descriptions[ko] = ko  # Use the KO term itself if the name can't be retrieved
    return ko_descriptions