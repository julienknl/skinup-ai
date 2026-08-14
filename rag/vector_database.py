import chromadb

class Database:

    def __init__(self, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="ingredient_knowledge")

    def add_documents(self, documents):
        """
        This function add the documents and metadata in the vector database.
        The parameter documents is a list of dictionaries which contain keys id, content and metadata.
        """
        try:
            self.collection.add(ids=[document["id"] for document in documents],
                documents=[document["content"] for document in documents],
                metadatas=[document["metadata"] for document in documents])
        except ValueError as e:
            print(f"Error: {e}")

    def retrieve(self, queries, n_results=1, where=None):
        """
        This function retrieve contents based from the input queries.
        The parameters consist of:
        - queries: The query texts.
        - n_results: The number of most likely results
        - where: A where clause to match a specific search e.g. retrieve content 
        specifically from a specific skin condition {condition : acne}.
        """
        try:
            results = self.collection.query(
                query_texts=queries,
                n_results=n_results,
                where=where
            )

            return results
        
        except Exception as e:
            print(f"Error: {e}")