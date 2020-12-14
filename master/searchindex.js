Search.setIndex({docnames:["explanations/performance","explanations/sans-io","how-to/library-hdf","how-to/poll-changes","index","reference/api","reference/appendix","reference/changelog","reference/contributing","tutorials/commandline-hdf","tutorials/control","tutorials/installation","tutorials/load-save"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["explanations/performance.rst","explanations/sans-io.rst","how-to/library-hdf.rst","how-to/poll-changes.rst","index.rst","reference/api.rst","reference/appendix.rst","reference/changelog.rst","reference/contributing.rst","tutorials/commandline-hdf.rst","tutorials/control.rst","tutorials/installation.rst","tutorials/load-save.rst"],objects:{"":{T:[6,0,1,""]},"pandablocks.asyncio":{AsyncioClient:[5,0,1,""]},"pandablocks.asyncio.AsyncioClient":{close:[5,2,1,""],connect:[5,2,1,""],data:[5,2,1,""],send:[5,2,1,""]},"pandablocks.blocking":{BlockingClient:[5,0,1,""]},"pandablocks.blocking.BlockingClient":{close:[5,2,1,""],connect:[5,2,1,""],data:[5,2,1,""],send:[5,2,1,""]},"pandablocks.commands":{Arm:[5,0,1,""],Command:[5,0,1,""],CommandException:[5,3,1,""],Get:[5,0,1,""],GetBlockNumbers:[5,0,1,""],GetChanges:[5,0,1,""],GetFields:[5,0,1,""],GetPcapBitsLabels:[5,0,1,""],Put:[5,0,1,""],Raw:[5,0,1,""]},"pandablocks.commands.Command":{lines:[5,2,1,""],ok_if:[5,2,1,""],response:[5,2,1,""]},"pandablocks.commands.Get":{response:[5,2,1,""]},"pandablocks.commands.GetBlockNumbers":{response:[5,2,1,""]},"pandablocks.commands.GetFields":{response:[5,2,1,""]},"pandablocks.commands.Raw":{response:[5,2,1,""]},"pandablocks.connections":{Buffer:[5,0,1,""],ControlConnection:[5,0,1,""],DataConnection:[5,0,1,""],NeedMoreData:[5,3,1,""]},"pandablocks.connections.Buffer":{peek_bytes:[5,2,1,""],read_bytes:[5,2,1,""],read_line:[5,2,1,""]},"pandablocks.connections.ControlConnection":{receive_bytes:[5,2,1,""],send:[5,2,1,""]},"pandablocks.connections.DataConnection":{connect:[5,2,1,""],flush:[5,2,1,""],receive_bytes:[5,2,1,""]},"pandablocks.hdf":{FrameProcessor:[5,0,1,""],HDFWriter:[5,0,1,""],Pipeline:[5,0,1,""],create_pipeline:[5,5,1,""],stop_pipeline:[5,5,1,""],write_hdf_files:[5,5,1,""]},"pandablocks.hdf.Pipeline":{stop:[5,2,1,""],what_to_do:[5,4,1,""]},"pandablocks.responses":{Data:[5,0,1,""],EndData:[5,0,1,""],EndReason:[5,0,1,""],FieldCapture:[5,0,1,""],FieldType:[5,0,1,""],FrameData:[5,0,1,""],ReadyData:[5,0,1,""],StartData:[5,0,1,""]},"pandablocks.responses.EndData":{reason:[5,4,1,""],samples:[5,4,1,""]},"pandablocks.responses.EndReason":{DATA_OVERRUN:[5,4,1,""],DISARMED:[5,4,1,""],DMA_DATA_ERROR:[5,4,1,""],DRIVER_DATA_OVERRUN:[5,4,1,""],EARLY_DISCONNECT:[5,4,1,""],FRAMING_ERROR:[5,4,1,""],OK:[5,4,1,""]},"pandablocks.responses.FieldCapture":{capture:[5,4,1,""],name:[5,4,1,""],offset:[5,4,1,""],scale:[5,4,1,""],type:[5,4,1,""],units:[5,4,1,""]},"pandablocks.responses.FieldType":{subtype:[5,4,1,""],type:[5,4,1,""]},"pandablocks.responses.FrameData":{column_names:[5,2,1,""],data:[5,4,1,""]},"pandablocks.responses.StartData":{fields:[5,4,1,""],format:[5,4,1,""],missed:[5,4,1,""],process:[5,4,1,""],sample_bytes:[5,4,1,""]},pandablocks:{asyncio:[5,1,0,"-"],blocking:[5,1,0,"-"],commands:[5,1,0,"-"],connections:[5,1,0,"-"],hdf:[5,1,0,"-"],responses:[5,1,0,"-"]},socket:{socket:[6,0,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","module","Python module"],"2":["py","method","Python method"],"3":["py","exception","Python exception"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:module","2":"py:method","3":"py:exception","4":"py:attribute","5":"py:function"},terms:{"100":[2,8],"1000":[2,5,9],"10000000":9,"1048576":5,"10m":9,"1hz":4,"1ms":9,"1us":9,"2020":4,"20s":9,"2ms":9,"30kbyte":9,"30mbyte":9,"40000000":2,"40s":2,"45mbyte":2,"50000000":4,"500hz":9,"50mbyte":0,"600mbyte":9,"602m":9,"60mbyte":0,"60s":2,"74k":9,"abstract":5,"break":2,"byte":[0,1,4,5,9],"class":[1,4,5,6],"default":[5,9],"final":[2,4],"float":[2,5],"function":[0,4],"import":[2,4,8,9],"int":[2,5],"new":[4,5,8,12],"return":[1,2,5,10],"short":5,"static":8,"switch":0,"true":[0,1,2,5],"try":2,"while":[0,1,5,8,9],DLS:11,For:[1,2,5,11],NFS:[0,9],One:2,The:[0,1,2,5,6,8,9,11],There:[0,2,5],These:[0,6,8],Using:0,With:2,about:[2,3,5],abov:[0,9],accept:0,access:5,accomplish:5,accord:5,achiev:4,acquisit:[2,4,5,9],action:[4,5],activ:[0,4,5,10,11],add:[9,12],adding:2,address:[0,9,10],adher:7,advantag:1,affect:0,after:[2,4,5,9],aid:1,all:[0,2,5,7,8,9,12],allow:[0,1,5],alphabet:5,also:[0,2,8,9,10,11],although:11,alwai:2,amount:5,analysi:0,ani:[1,5,8,10,11],api:[1,4,8],applic:[2,9],approach:[0,4],arg:12,argument:[0,1],argv:[2,9],arm:[2,5,9,10],arm_and_hdf:2,arrai:5,arrow:10,articl:0,assembl:5,assert:2,assum:[9,12],async:[1,2,5],asyncgener:5,asyncio:[1,2,4],asynciocli:[0,1,2,5],attribut:5,automat:8,avail:[0,1],averag:[4,5],await:[1,2,5],back:5,bandwidth:5,bar:2,base:[1,2,5],baseclass:5,basic:12,basicconfig:2,becaus:2,been:[5,11],befor:[2,5,8,9],begin:[2,5],being:[5,9],benefit:[0,5],best:1,better:1,big:8,bigger:9,biggerfil:9,bin:11,bit:[5,9],bit_mux:5,bit_out:5,bits0:[4,10],bits1:[4,10],bits2:[4,10],bits3:[4,10],black:8,blink:12,block:[0,1,2,4,9,12],blockingcli:[0,1,4,5],bool:5,both:[0,1,2,4,5],browser:0,buf:5,buffer:5,bug:8,build:8,bytearrai:5,bytes_from_serv:5,call:[1,4,5],callabl:5,can:[1,2,4,5,8,9,10,11,12],captur:[0,4,5],carriag:2,caught:8,chang:[4,5,8,9],check:[2,4,8,9,10],choos:9,chunk:5,cli:12,client:[1,2,8,9,10,11],clone:8,close:[2,4,5,9],cmd1:[1,5],cmd2:[1,5],cmd3:5,code:[1,2,4,5,9],collect:[4,5],column:5,column_nam:5,com:[4,8,11],combin:0,command:[1,2,4,6,9],commandexcept:5,commandlin:[0,2,4,5,10,11],common:5,commun:5,complet:[2,5,9,10],compliant:9,compon:5,concept:12,conclus:4,concurr:1,condit:0,configur:[2,5,9],conform:8,congest:5,connect:[0,2,4,9],consid:[2,4],consol:4,consum:5,contain:[5,8],content:9,context:2,contribut:4,control:[1,2,4,5,9],controlconnect:[1,5],conveni:5,convent:8,core:[0,2,4],correspond:[1,5],could:2,counter1:5,counter2:5,counter:9,cours:1,cover:9,coverag:8,cpu:[0,5],creat:[4,5],create_pipelin:[2,5],current:[5,10,11,12],custom:2,data:[1,2,4,5],data_overrun:[0,5],dataconnect:[1,5],dataset:[4,5,9],datatyp:0,decis:2,def:2,definit:[5,6],deliveri:5,demo:12,demonstr:12,depend:[0,1],descript:5,detail:8,detect:5,dev:8,dict:[1,5],dictionari:5,differ:1,directli:4,directori:8,disarm:[4,5,10],disconnect:5,discuss:0,disk:[0,5,9],displai:[2,10],distribut:12,dls:11,dma:5,dma_data_error:5,do_something_with:[1,5],doc:[6,8],docstr:8,document:[7,11,12],doe:[3,8],doing:9,don:8,done:[2,5],down:[2,5],downstream:5,drawn:2,driver:5,driver_data_overrun:5,drop:0,dtype:5,each:[0,2,5,9],earli:5,early_disconnect:5,easili:8,edg:5,effici:[0,4,5],either:1,element:[2,5],elif:2,emit:2,empti:5,enabl:[4,5,10],encapsul:1,end:[2,4,5,9],enddata:[2,5],endreason:[2,5],enough:[0,5],entri:[],environ:4,error:[5,8],establish:5,etc:5,ethernet:0,event:5,everi:5,examin:4,exampl:[1,2,5,9],except:5,excess:0,execut:[1,5],exist:[8,11],expens:1,experi:5,experienc:4,expir:0,explicitli:1,explor:9,expos:[1,5],extern:11,extra:0,extract:8,factor:[4,5],fail:0,fall:5,fals:[1,2,5],fast:[2,4,5,9],faster:4,favourit:9,fdata:5,featur:[1,4,11],feed:[1,2],field:[0,1,2,5,9,10,12],fieldcaptur:5,fieldtyp:[1,5],file:[4,5,7,8,11,12],filenam:5,filepath:5,filesystem:9,firefox:8,first:[1,2,4,9],fit:8,fix:8,flag:12,flake8:8,flow:2,flush:[1,4,5],flush_every_fram:[1,5],flush_period:[0,1,2,5],flushabl:[1,5],follow:[0,8,9],form:5,format:[0,5,8],fpga:[0,12],fraction:2,frame:[0,2,5,9],frame_timeout:5,framedata:[1,2,5],frameprocessor:[2,5],framework:[1,2],framing_error:5,free:8,frequenc:0,from:[1,2,4,5,8,9,10,11,12],from_serv:[1,5],full:5,gate:[4,10],gather:[1,2,5],gener:[0,6],get:[0,1,2,4,5,8,9,10],getblocknumb:5,getchang:5,getfield:[1,5],getpcapbitslabel:5,gigabit:0,gil:5,git:[8,11],github:[4,8,11],give:[0,2,5],given:0,got:[2,4],gpf:9,great:8,greatest:0,guarante:1,gui:[0,4],guid:2,h5diff:9,h5py:[5,9,11],had:1,handl:[5,8],handler:5,has:[1,5,9,10,11],have:[2,5,8,9],hdf5:[4,9],hdf:[4,11],hdf_queue_report:2,hdfwriter:[2,5],head:8,headl:8,health:[4,10],heavi:5,hello:4,help:4,helper:5,henc:9,here:[4,6],high:[5,9],higher:0,hire:9,hit:[4,5,10],host:5,hostnam:[1,4,5,9,10],how:[5,8,9,10,12],html:[4,5,8],http:[4,5],idea:8,idn:[4,5],implement:5,improv:[0,8],includ:[1,5],increas:[0,9],index:[4,8],indic:6,info:[2,4,9],inform:[0,5],inherit:5,initi:[1,7],inp:5,inpa:5,input:5,insid:11,inspect:2,instal:[0,4,8],instanc:[1,2],instruct:11,integr:[1,2,9],intel:0,intend:11,interact:4,interfac:[1,5,9,11],interfer:11,intermedi:[0,1],intern:[5,11],introduc:2,involv:8,isinst:2,isn:5,isort:8,issu:8,iter:[1,2,5],its:[5,6],join:5,just:9,kei:[4,10],know:[9,12],label:[5,9],larger:0,last:5,late:5,later:[9,11],latest:5,launch:9,led:12,legend:9,len:2,let:9,level:[2,5,9],librari:[0,4,5,8,9],lift:5,like:[2,3,5,9,12],limit:5,line:[4,5],linux:10,list:[1,5,9],listen:[2,9],live:5,load:[0,4,9],local:[0,9],locat:9,log:[2,4],logic:5,look:[1,5,9,12],low:9,lower:0,lut:5,machin:0,mai:0,make:[1,2,8,9],mal:5,malcolm:3,manag:2,mani:[0,5],manual:5,map:1,markup:5,master:4,match:5,materi:4,matplotlib:9,max:0,maximis:0,maximum:[0,9],mean:[0,2,9,10],measur:0,memori:5,messag:2,method:[1,4],might:[8,12],million:9,min:0,miss:5,mode:[0,9],modest:9,modul:[5,12],more:[0,1,2,4],most:[0,5,8],mount:[0,9],much:1,multi:5,multilin:5,multipl:[1,9],mypi:8,name:[1,5,9],nativ:9,nclose:2,ndarrai:5,need:[1,2,5,6,11],needmoredata:5,network:[0,5],newlin:[2,5],next:12,none:5,notabl:7,noth:7,now:[5,10,11],num:[5,9],number:[0,1,5,8],numpi:[0,5],object:[1,2,5],occur:5,off:[1,5],offset:5,often:[0,5],ok_if:5,onc:5,one:[2,5,8,9],oneshot:5,onli:[0,1,5],open:[4,9,10],optimis:2,option:[1,5],order:[5,8],other:[1,9,10],ourselv:2,out:[5,9],outlin:[2,12],output:5,over:[2,5],overflow:5,overload:5,overrun:5,own:5,packag:[4,5],page:8,panda:[1,2,4,5,9,10,12],pandablock:[0,1,2,5,8,9,10,11,12],param:5,paramet:[5,6],pariti:1,pars:5,partial:[5,9],pass:[1,2,5],past:2,path:11,payload:0,pcap:[4,5,9,10],pcapbitslabel:5,pdf:9,peek_byt:5,per:5,perform:[4,9],period:[0,9],pip:[4,11],pipelin:[4,5],pipenv:[8,11],pleas:[8,11],plot:9,plot_counter_hdf:9,plt:9,png:9,poll:4,pop:5,port:[1,2,4,5,9,12],pos_mux:5,posix:9,possibl:[5,9],practic:4,prescal:[2,9],present:10,press:10,previou:10,print:[2,4,9],print_progress_bar:2,probabl:5,process:[0,1,5],produc:[0,9],product:9,program:1,progress:[2,10],project:[4,7,8],prompt:10,properti:5,protocol:1,provid:[0,5],publish:4,pull:8,pulse1:5,put:[1,2,5],put_nowait:2,pypi:4,pyplot:9,python3:11,python:2,quell:6,queri:10,queu:2,queue:[2,5],quickli:5,rais:5,rang:9,rate:[0,9],rather:[0,2],raw:[0,5],reach:5,read:[5,8,9],read_byt:5,read_lin:5,reader:[5,9],readi:[2,5],readthedoc:5,readydata:[2,5],reason:[2,4,5,9],recal:10,receiv:[1,5],receive_byt:[1,5],recommend:11,recv:[1,5],reduc:[0,5,8,9],refer:1,releas:[7,11],remain:8,remov:4,repeat:[2,9],repeatedli:[1,5],replac:5,repo:12,repons:5,report:8,repositori:[8,9],request:[5,8],requir:[1,9,11],resp1:[1,5],resp2:[1,5],resp3:5,respect:1,respond:5,respons:[1,2,4,6,10],result:0,reusabl:1,routin:[1,2],row:5,run:[0,2,4,5,9],sai:[1,5],same:[0,1,2,8,9],sampl:[0,4,5,9,10],sample_byt:5,san:[4,5],save:[4,9],scalar:5,scale:[2,4,5],scan:[0,5],scheme:[2,5],scope:8,screenshot:12,second:[1,2,9],see:[9,11],select:0,semant:7,send:[0,1,2,4,5,10],sendal:[1,5],sent:[0,1,4,5],separ:[1,5],seq1:[2,5,9],seq:9,server:[0,1,4,5,10,12],set:[0,2,8,9,10],setup:9,shift_sum:[4,10],shot:2,should:[0,1,2,5,9,10,11,12],show:[2,9,10,12],shown:[9,10],side:0,significantli:8,similar:9,simpl:[2,5],simplic:1,simultan:2,sinc:5,singl:[0,1,2,5,9],size:9,slightli:1,slow:0,socket:[1,5,6],softwar:11,some:[0,1,2,4,5],someth:[3,8],soon:5,sort:5,sourc:[1,4,5,8,9,11],special:10,specif:5,speed:0,spend:8,sphinx:6,squash:[0,1],ssd:0,stabil:0,standard:8,star:[5,10],start:[2,4,5,9,10],startdata:[0,5],state:12,step:[2,4],still:0,stop:[2,5,10],stop_pipelin:[2,5],storag:5,store:[4,5,9],str:[1,5],strategi:[4,9],string:[1,5],structur:5,style:4,subclass:[1,5],subsequ:2,subtyp:5,success:5,suitabl:5,support:[1,5],sure:8,sustain:[0,9],swmr:9,sys:[2,9],system:0,tab:[4,10],tabl:5,take:[1,2,5],tax:9,tcp:[0,1,4,5,12],technic:4,tell:[2,5,9],termin:[5,9,10,11],test:[0,4,9],than:[0,2],thei:[1,5,9,11],them:[0,1,5],thi:[0,1,2,4,5,6,7,8,9,10,11,12],thread:5,three:9,through:[2,8],throughput:[0,5],tick:9,ticket:8,time:[0,1,2,5,8,9],timeout:5,timeouterror:5,titl:8,tmp:[2,4,9],to_send:[1,5],todo:4,togeth:[0,1],too:5,took:2,tool:[4,8,9,10,12],top:[2,5],total:[5,9],transfer:0,transform:5,transmit:5,trig:[4,5,10],trig_edg:[4,10],trigger:[0,5],ts_end:[4,10],ts_start:[4,10],ts_trig:[4,10],tupl:[1,5],tutori:2,twenti:9,twice:9,type:[0,1,2,4,5,6,8,11],uint:5,underlin:8,union:5,unit:[5,8,9],unreleas:4,unscal:0,until:0,updat:2,usabl:1,usag:[0,4],use:[0,1,4,5,9,10,11,12],used:[0,5,10],useless:1,user:4,uses:[0,5],using:[3,9],util:0,valu:[0,1,5,9,10],variabl:10,venv:11,verbos:1,veri:9,version:[0,4,7],view:5,virtual:4,volum:0,wai:[4,5],wait:[1,2,5],want:[0,2,5,8],warn:6,web:[0,4],webcontrol:4,welcom:8,well:9,were:[0,5],what:[0,1,5,12],what_to_do:5,when:[0,1,2,5,8,9],whenev:1,where:[5,9,10],whether:5,which:[1,2,5,9],why:4,window:9,wire:[1,5],without:[1,5],work:[4,11],would:1,wrapper:[4,5],write:[2,3,4,9,11],write_hdf_fil:[0,2,5],writer:9,written:[0,5,9],yaml:4,yet:7,yield:5,you:[0,2,5,8,9,10,11,12],your:[4,8,9,10],yourself:4,zpkg:0},titles:["How fast can we write HDF files?","Why write a Sans-IO library?","How to use the library to capture HDF files","How to efficiently poll for changes","PandABlocks Python Client","API","Appendix","Change Log","Contributing","Commandline Capture of HDF Files Tutorial","Interactive Control Tutorial","Installation Tutorial","Commandline Load/Save Tutorial"],titleterms:{"1hz":0,"2020":7,"function":2,about:[0,4],achiev:0,acquisit:10,api:5,appendix:6,approach:2,asyncio:5,averag:0,block:5,call:2,can:0,captur:[2,9],chang:[3,7],check:11,client:[0,4,5],code:8,collect:9,command:[5,10],commandlin:[9,12],conclus:[9,10],connect:[1,5,10],consid:0,contribut:8,control:10,creat:[2,11],data:[0,9],directli:2,document:[4,8],effici:3,environ:11,examin:9,explan:4,factor:0,fast:0,faster:9,file:[0,2,9],flush:0,gui:12,guid:4,hdf:[0,2,5,9],help:0,how:[0,2,3,4],instal:11,interact:10,librari:[1,2,11],load:12,log:7,more:9,packag:0,panda:0,pandablock:4,perform:[0,2],pipelin:2,poll:3,python:[4,11],refer:4,remov:0,respons:5,run:8,san:1,save:12,scale:0,some:9,strategi:0,structur:4,style:8,test:8,tutori:[4,9,10,11,12],type:10,unreleas:7,use:2,version:11,virtual:11,web:12,webcontrol:0,why:1,wrapper:1,write:[0,1,5],your:11,yourself:2}})