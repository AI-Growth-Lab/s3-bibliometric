\name{s3_triples}
\alias{s3_triples}
\alias{s3_add_triples}
\alias{s3_extract_triples}
\title{Knowledge-graph triples: methods, data, theories, settings and findings}
\description{
Triples link a paper to what it uses or studies (\code{USES_METHOD},
\code{USES_DATA}, \code{USES_THEORY}, \code{STUDIES}, \code{STUDIED_IN},
\code{IN_INDUSTRY}, \code{AT_LEVEL}) or one construct to another as a finding
(\code{INCREASES}, \code{DECREASES}, \code{NO_EFFECT_ON}, \code{MODERATES},
\code{MEDIATES}, \code{ASSOCIATED_WITH}). They join S3's knowledge graph with the
themes, authors and journals, and answer Q39 and Q40.

\code{s3_add_triples()} adds triples from a file or data frame.
\code{s3_extract_triples()} extracts them from the titles and abstracts with a
language model, batch by batch with a progress message after each batch:
\code{"claude-code"} uses your Claude plan through the Claude Code command line,
\code{"codex"} your ChatGPT plan through the Codex command line, and
\code{"anthropic"} or \code{"openai"} an API key in \env{ANTHROPIC_API_KEY} or
\env{OPENAI_API_KEY}. Answers are cached, so an interrupted run resumes.
\code{s3_triples()} returns the run's triples.
}
\usage{
s3_add_triples(res, triples, replace = FALSE)
s3_extract_triples(res, backend = c("claude-code", "codex", "anthropic", "openai"),
                   model = NULL, batch_size = 15L, max_papers = NULL, quiet = FALSE)
s3_triples(res)
}
\arguments{
  \item{res}{An S3 result.}
  \item{triples}{A CSV, Excel or JSON file, or a data frame, with columns
    \code{paper_id} (EID, DOI or title), \code{subject}, \code{relation} and
    \code{object}; optionally \code{evidence}.}
  \item{replace}{Replace the run's triples instead of adding to them.}
  \item{backend}{Which model extracts the triples.}
  \item{model}{Model name; optional except for \code{"openai"}.}
  \item{batch_size}{Papers per request.}
  \item{max_papers}{Only the first papers, e.g. to test.}
  \item{quiet}{Hide progress messages.}
}
\value{\code{s3_add_triples()} and \code{s3_extract_triples()}: a list with the
  number of triples and papers matched and anything left out.
  \code{s3_triples()}: a data frame.}
\examples{
\dontrun{
res <- s3_run(s3_example())
s3_extract_triples(res, "claude-code", max_papers = 30)
s3_answer(res, "Q39")
}
}
