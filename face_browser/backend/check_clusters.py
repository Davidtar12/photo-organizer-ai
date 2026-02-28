from database import session_scope
from models import PetCluster, PetEmbedding
from sqlalchemy import func

with session_scope() as s:
    clusters = s.query(PetCluster.id, PetCluster.species, func.count(PetEmbedding.id).label("count")) \
        .outerjoin(PetEmbedding, PetEmbedding.cluster_id == PetCluster.id) \
        .group_by(PetCluster.id) \
        .all()
    
    print(f"Total PetCluster rows: {len(clusters)}")
    for cid, species, count in clusters:
        print(f"  Cluster {cid}: species={species}, count={count}")
