from rag.vector_database import Database

class RetrieveKnowledge:

    def __init__(self, database):
        self.database = database

    def retrieve(self, queries, n_results=1, where=None):
        return self.database.retrieve(
            queries=queries,
            n_results=n_results,
            where=where
        )

if __name__ == "__main__":

    database = Database()
    skin_condition = "oily"

    retrieval = RetrieveKnowledge(database=database)
    results = retrieval.retrieve(queries=[f"ingredients that help having a skin condition {skin_condition}"],
                       n_results=1,
                       where={"condition" : skin_condition})
    print(results)