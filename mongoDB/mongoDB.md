# MongoDB 

## 1. database 
- 데이터베이스는 컬렉션의 물리적 컨테이너. 하나의 데이터베이스에는 보통 여러개의 컬렉션을 갖고 있음

## 2. collection
- mongoDB Document의 그룹이며, RDMBS의 Table 개념과 비슷
- 단일 데이터베이스에 존재
- 스키마를 강요하지 않음. 따라서 컬렉션 내부의 도큐먼트는 서로 다른 필드를 가질 수 있음
- 도큐먼트는 일반적으로 서로 유사하거나 관련된 목적이 있음 

## 3. Document
- 하나의 키와 값의 집합

<br/>

|RDBMS|MongoDB|
|------|---|
|database|database|
|collection|table|
|document|row|
|column|field|
|index|index|
|dbserver|mongod|
|dbclient|mongo|

<br/>

## 4. 기본쿼리 
- 데이터 삽입 : db.users.insetOne( json 형태로 삽입)
- 데이터 찾기 : db.users.find( {key : value})
- 데이터 수정 : db.users.updateMany({key : value})
- 데이터 삭제 : db.users.deleteMany({key: value})

* mongoDB는 인덱스가 없다면 full scan 방식을 사용. 또한 콜렉션 생성시 _id 필드를 생성하지 않으면 자동으로 id 필드 object id값 설정 (중복방지)


## 5. GridFS
- 파일을 단일 문서에 저장하는 대신 파일을 청크로 나누고 각 청크를 별도의 문서로 저장. 마지막 청크를 제외하고 244kB 크기로 나눔. (기본 청크 GridFS : 255 kB)
- 파일 청크(fs.files)와 파일 메타데이터(fs.chunks) 두 내용을 저장
<br/>
  1. GridFS 컬렉션 : chunks
    - 바이너리 청크를 저장 
    ```
        {
        "_id" : <ObjectId>,
        "files_id" : <ObjectId>,
        "n" : <num>,
        "data" : <binary>
        }

    ```
    - _id : 청크의 고유 ObjectId
    - files_id : _id 컬렉션에 명시된 상위 문서의 files
    - n : 청크의 시퀸스 번호 (0부터 시작)
    - data : BSON Binary 유형

<br/>
  2. GridFS 컬렉션 : files
    - 파일의 메타데이터 저장
    ```
        {
        "_id" : <ObjectId>,
        "length" : <num>,
        "chunkSize" : <num>,
        "uploadDate" : <timestamp>,
        "md5" : <hash>,
        "filename" : <string>,
        "contentType" : <string>,
        "aliases" : <string array>,
        "metadata" : <any>
        }

    ```
    - _id : 고유 식별자. MongoDB 문서의 기본 유형은 BSONObjectId
    - length : 문서의 크기
    - chunkSize : 각 청크의 크기(바이트). 기본 크기는 255 킬로바이트(kB)
    - uploadDate : 문서가 GridFS에 처음 저장된 날짜. Date 유형
    - filename : 사람이 읽을 수 있는 GridFS 파일의 이름
    - metadata : 선택사항
