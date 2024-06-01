import pysolr
solr_url = 'http://localhost:8983/solr'
solr = pysolr.Solr(solr_url , always_commit=True)

solr.ping()
