generator target {
  output_type "python/pydantic"
  output_dir "../baml_client"
  version "0.225.0"
  default_client_mode sync
}

function Hello(message: string) -> string {
  client "openai/gpt-4o-mini"
  prompt #"
    {{ _.role("user") }}
    {{ message }}
  "#
}
