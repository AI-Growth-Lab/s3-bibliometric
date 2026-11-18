\name{s3_communities}
\alias{s3_communities}
\alias{s3_papers}
\alias{s3_network_metrics}
\alias{s3_node_metrics}
\alias{s3_scan}
\alias{s3_search}
\title{Results as data frames}
\description{
\code{s3_communities()} lists the communities with their size, years,
citations, QI and MI. \code{s3_papers()} gives one row per paper with its
community, centrality and outlier flag. \code{s3_scan()} is the threshold scan
(silhouette and SII per threshold). \code{s3_search()} finds the papers closest
in meaning to a query. \code{s3_network_metrics()} gives the network-level indicators of question Q41
(one row per indicator, one column per network: the map and, at level 2, the
citation and co-authorship layers); \code{s3_node_metrics()} gives one row per
paper with its degree, strength, eigenvector, betweenness, harmonic closeness,
PageRank, clustering, k-core, participation, within-community z and role
(hub, connector or peripheral).
}
\usage{
s3_communities(res, min_size = 2L)
s3_papers(res, text = FALSE)
s3_scan(res)
s3_network_metrics(res, layers = NULL)
s3_node_metrics(res)
s3_search(res, query, top_k = 10L)
}
\arguments{
  \item{res}{An S3 result from \code{\link{s3_run}} or \code{\link{s3_load}}.}
  \item{min_size}{Smallest community to list.}
  \item{text}{Keep the combined title and abstract column.}
  \item{query}{Search text.}
  \item{top_k}{Number of papers to return.}
  \item{layers}{Include the citation and co-authorship layers (\code{NULL}: yes at level 2).}
}
\value{A data frame.}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_communities(res)
s3_node_metrics(res)
s3_search(res, "absorptive capacity and innovation")
}
}
