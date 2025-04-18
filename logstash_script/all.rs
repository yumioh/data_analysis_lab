input{
    http_poller {
       urls => {
           cat_api => {
              method => get
              url => "https://catfact.ninja/fact"
          }
        }
        tags => "cat_api"
        request_timeout => 100
        schedule => { every =>"1m"}
        codec => json
        metadata_target => "http_poller_metadata"
   }
    http_poller {
      urls => {
        seoul_safety_api => {
            method => get
            url => "http://openapi.seoul.go.kr:8088/575869646a696f763639596d5a7058/xml/pmisSafetyCheckM/1/5/043012061101"
         }
      }
      tags => "seoul_safety_api"
      request_timeout => 100
      schedule => { every =>"5m"}
      codec => plain
      metadata_target => "http_poller_metadata"
    }
    http_poller {
      urls => {
        kosha_api => {
          method => get
          url => "https://apis.data.go.kr/B552468/disaster_api01/getdisaster_api"
          headers => {
            Accept => "application/json"
          }
          query => {
            serviceKey => "6KKekmgnVsoZy3Sr/l/rlXBCT+eZ2t9YgsUBXdb93wGbF6OiwoGiHOjBFgux/i2Yet5A5FuLwm77QOUmYdgYww==" # 디코딩한 인증키값을 넣어야함
            pageNo => "2"
            numOfRows => "20"
            business => "건설업"
          }
        }
      }
      tags => "kosha_api"
      request_timeoutㄴㅍㅍ => 60
      schedule => { every => "1m" } 
      codec => json
    }
    http_poller {
      urls => {
        seoul_bizinfo_api => {
            method => get
            url => "http://openAPI.seoul.go.kr:8088/575869646a696f763639596d5a7058/xml/ListOnePMISBizInfo/1/5"
         }
      }
      tags => "seoul_bizinfo_api"
      request_timeout => 100
      schedule => { every =>"5m"}
      codec => plain
      metadata_target => "http_poller_metadata"
    }
  }
  filter {
    if "seoul_safety_api" in [tags] or "seoul_bizinfo_api" in [tags] {
      xml {
        source => "message"
        target => "parsed_xml"
        force_array => false
      }
      split { 
        field => "[parsed_xml][row]"
      }
      mutate {
        rename => {"[parsed_xml][row]" => row}
        remove_field => "[parsed_xml][RESULT]"
      }
    }
    if "cat_api" in [tags] {
      mutate {
        split => { "fact" => " " }
      }
    }
    date {
      match => ["log_time", "yyyy-MM-dd HH:mm:ss"]
      target => "@timestamp"
    }
    mutate {
      add_field => { 
        "log_time" => "%{+YYYY-MM-dd HH:mm:ss}" 
      }
      remove_field => [
        "[http_poller_metadata]",
        "@version",
        "[event][original]"
      ]
    }
  }
  output {
    if "cat_api" in [tags] {
      elasticsearch{
            hosts => ["http://192.168.0.4:9200"]
            index => "cat_api"
      }
    }
  
    if "seoul_safety_api" in [tags] {
      elasticsearch{
          hosts => ["http://192.168.0.4:9200"]
          index => "seoul_safety_api"
      }
    }
    if "kosha_api" in [tags] {
      elasticsearch{
          hosts => ["http://192.168.0.4:9200"]
          index => "kosha_api"
      }
    }
    if "seoul_bizinfo_api" in [tags] {
      elasticsearch{
          hosts => ["http://192.168.0.4:9200"]
          index => "seoul_bizinfo_api"
      }
    }
    stdout {
      codec => rubydebug
    }
  }