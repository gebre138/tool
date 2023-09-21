get_ipython().system('pip install amseg')
get_ipython().system('pip install glob')
get_ipython().system("pip install '/content/drive/MyDrive/bert/tool/HornMorphoA-4.3-py3-none-any.whl'")

try:
    from IPython.display import clear_output
    clear_output(wait=True)
except ImportError as err:
    print(err)
try:
    import os
    import pandas as pd
    import glob
    import re
    import hm
    import numpy as np
    from termcolor import colored
    from gensim.models import Word2Vec, KeyedVectors   
    from keras.models import load_model
    from collections import Counter
    from amseg. amharicNormalizer import AmharicNormalizer as normalizer
except ImportError as err:
    print(err)

def ComplexityAnotator(text):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    path = os.getcwd()
    paths=""
    if path=="/content":
        paths="/content/drive/MyDrive/"
    else:
        paths=path
    if not os.path.exists(paths+"/dataset"):
        os.makedirs(paths+"/dataset")
    if not os.path.exists(paths+"/dataset/stoplist"):
        os.makedirs(paths+"/dataset/stoplist") 
    if not os.path.exists(paths+"/dataset/other"):
        os.makedirs(paths+"/dataset/other") 

    if not os.path.exists(paths+"dataset/stoplist/spchar.txt"):
        spch="፩ ፫ ፪ ፬ ፭ ፮ ፯ ፰ ፱ ፲ ፳ ፴ ፵ ፶ ፷ ፸ ፹ ፶ ፺ ፯ ፻ ፼ 0 1 2 3 4 5 6 7 8 9 { } a A b B c C d D e E f F g G h H i I j J k K l L m M \
        n N o O p P q Q r R s S t T u U v V w W x X y Y z Z '"' | :\ ; , . / < > ? [ ] ; , . / ፤ ፣ ። ፡  ” “ ፠ ፥ ፦ ፧ ፨ \ ´ … \
        !「 "'" ¦ _ , \ ¨ ፣ ፤ . ፹ ። ~ ! @ # $ % ^ & * ( ) _ + ` ፟ = - – \ufeff • ★ 🙂 � "
        spch = spch.split()
        with open(paths+'/dataset/stoplist/spchar.txt', 'a',encoding="utf-8") as file:
            for i in spch:
                file.write(i+"\n")

    if not os.path.exists(paths+"/dataset/stoplist/amharic_stop_lists.txt"):
        amstop="የ ለ በዚህ እንደ ነገር አንድ አንድን እና አለ አየ የት ግኝ በላ ሆነ ለየ ባለ ጊዜ ሄደ በ አል ሃ ያ ጋ ሆነ ነገረ ነበረ ወይም ሆኑ ሆኖም ነው ናቸው ነበር ሁሉንም ላይ ሌላ ሌሎች ስለ \
        ቢሆን ብቻ መሆኑ ማለት ማለቱ የሚገኝ የሚገኙ ማድረግ ማን ማንም ሲሆን ሲል እዚህ እንጂ በኩል በውስጥ በጣም ይህን በተለይ እያንዳንድ በሆነ ከዚህ ከላይ ከመሀል ከመካከል ከጋራ ጋራ ወዘተ \
        ወደ ያለ ሲሉ በተመለከተ በተመሳሳይ ያሉ የኋላ የሰሞኑ  ሁሉ ሁሉም ኋላ ሁኔታ ሆነ ሆኑ ሆኖም ሁል ሁሉንም ላይ ሌላ ሌሎች ልዩ መሆኑ ማለት ማለቱ መካከል የሚገኙ የሚገኝ ማድረግ ማን \
        ማንም ሰሞኑን ሲሆን ሲል ሲሉ ስለ ቢቢሲ ቢሆን ብለዋል ብቻ ብዛት ብዙ ቦታ በርካታ በሰሞኑ በታች በኋላ እባክህ በኩል በውስጥ በጣም ብቻ በተለይ በተመለከተ በተመሳሳይ የተለያየ የተለያዩ \
        ተባለ ተገለጸ ተገልጿል ተጨማሪ ተከናውኗል ችግር ታች ትናንት ነበረች ነበሩ ነበረ ነው ነይ ነገር ነገሮች ናት ናቸው አሁን አለ አስታወቀ አስታውቀዋል አስታውሰዋል እስካሁን አሳሰበ አሳስበዋል \
        አስፈላጊ አስገነዘቡ አስገንዝበዋል አብራርተዋል እባክዎ አንድ አንጻር እስኪደርስ እንኳ እስከ እዚሁ እና እንደ እንደገለጹት እንደተገለጸው እንደተናገሩት እንደአስረዱት እንደገና ወቅት እንዲሁም \
        እንጂ እዚህ እዚያ እያንዳንዱ እያንዳንዳችው እያንዳንዷ ከ ከኋላ ከላይ ከመካከል ከሰሞኑ ከታች ከውስጥ ከጋራ ከፊት ወዘተ ወይም ወደ ወደፊት ውስጥ እባክሸ ውጪ ያለ ያሉ ይገባል የኋላ የሰሞኑ \
        የታች የውስጥ የጋራ ያ ይታወሳል ይህ ደግሞ ድረስ ጋራ ግን ገልጿል ገልጸዋል ግዜ ጥቂት ፊት ደግሞ ዛሬ ጋር ተናግረዋል የገለጹት ይገልጻል ሲሉ ብለዋል ስለሆነ አቶ ሆኖም መግለጹን አመልክተዋል \
        ይናገራሉ አበራርተው አስረድተዋል እስከ ይህ ከነ ያለ ወደ ስለ ተራ ሙሉ ጋር እና ነው ግን ወይም እንጅ እንኳ ናቸው አዎን እንዲህ እነዚህ ምን ይኸውም"
        amstop = amstop.split()
        with open(paths+'/dataset/stoplist/amharic_stop_lists.txt', 'a',encoding="utf-8") as file:
            for i in amstop:
                file.write(i+"\n")
    spch=open(paths+"/dataset/stoplist/spchar.txt",'r',encoding="utf-8").read().split()
    amharicstop=open(paths+"/dataset/stoplist/amharic_stop_lists.txt",'r',encoding="utf-8").read().split()
    
    sentences=re.split('[?።!\n]', text)
    for i in sentences:
        if len(i)<250 and len(i)>10:
            with open(paths+'/dataset/sentence.txt', 'a',encoding="utf-8") as file:
                file.write(i+"\n")
    if not os.path.exists(paths+"/dataset/other/complex_word.xlsx"):
        root="ኮስማና መቃር ወደብ ቆረፈደ ለምጽ ደቦ ለሰሰ ለሰነ ለሴ ለቆታ ለበቅ ለበን ለበደ ለተተ ለከት ለኰፈ ለዘለዘ ለዘዘ ለገመ ለፈፈ ልባብ መነደገ ሞገድ ሞግዚት መሞጨጭ ረመጥ ረቀቀ \
        ረብጣ መቃቃር እንጭጭ ገነነ ርደት ርስት ርቱእ አርእስት ሮቄ ሰለቀ ሰለበ ሰለጠ ሰላጤ ሰረነቀ መስረግ መስረጽ ሰቀጠ ሰበቀ ተሰናሰለ ሰነበጠ ሰንበር ሴራ ስድ ሰጋር ሰገነት ሰግዳዳ ረባዳ ሲራክ \
        ሲሳይ ሳንቃ ባይተዋር ሴሰኛ ለቀለቀ ላሸቀ ወረት ቅይጥ ጋሬጣ መማስ ገረረ ኡደት ፍይዝ አሸለበ ሸለተ ሸመቀ ሸቀጠ ሸፈጠ ቀለሰ ቀሸረ ከሸነ ቀተረ ቀሰረ ቀነጠሰ ቀነጨረ ቀኖና ለበጠ ደጎሰ ጎራ \
        መስክ መረን ሰየመ ተሾመ ቋሳ ባተሌ ባዝራ ተሰበጣጠረ ነፈገ ተደራሲ ምናብ አደመ ንዋይ ንፉግ አለመ አለባ ማመንዠክ አሚካላ አምባረቀ መነዘረ መጥኔ ሀሩር አምባ ቅርፊት ደለል ክምችት ረቂቅ \
        ጥቅጥቅ እንቁ አቀበ ለደፈ ቋት አከንፋሽ ጎድፋሪ ሻቀለ ሸከነ ወጣራ ሰዋራ አሞካሸ ተመመ ግግ አደብ ኩታገጠም ወረዛ ወበቅ ደጓሳ ጅብራ ጥገት ፋይዳ ሀገረሰብ ሁለንተና ሁነኛ ሃሌታ ሀመልማል \
        ይስሙላ ሉአላዊ ሎጋ ሉጤ ሐራጅ ሀሴት መሳ ማእረግ ምንዝር ቃንዛ በለተተ ቡፍና ቦረቀ ቧልት መጋየት ባላንጣ ቆረቆሰ እልባት ተግሳጽ ተጠናወተ ቸልተኝነት ቸነፈር ነሸጠ ነውጥ ነፀብራቅ ረበረበ \
        አረፈቀ አርምሞ አተመ አዘቦት አገና አግቦኛ አጭልግ ኢምንት እቅጩን አከተመ እውን እንቦቀቅላ ከወነ ወረግቡ ዚቀኛ ዛበረ ዞማ ይፋ ደረመን ድድር ጀሌ ገርጃፋ ጎነቆለ ተጎሳቆለ ጥሞና ጨረፍታ \
        ጭብጥ ጸዳል ጽልመት ወና ወማ ጽጌ ፈተተ ፈገመ ፈገጠ ፍርኩታ ጎሬ ፎጠና ሀጫ ሀኬት ሁዳድ ህላዌ ህቡዕ ለውላዋ ለሆሳስ ለቆጠ ሌጣ ልጨኛ ልቀት ሎጥ መሰሪ መቃኝ መረኀ መቅኖ መነዘለ \
        ሙዳ መርግ መድብል መበለት ሙና ምጸት ማርጣ ማፍዳ መየሰ ሟጨ ረበበ ረብ መጫት ራደ ስናዳሪ ግዞት ተሰነቀ ሰነቀ ድብኝት ቀረረ ከርስ አምድ ሀሳዊ ቀጋ ሹም ሃሳዊ ሽብልቅ ሸነቆጠ \
        ሾመጠረ ሻጉራ ሾተል ሸፍጥ ቀልብ ቅምጥል ቀንጃ ቀበኛ ቀየደ ቁንዳላ ቁንጽል ውርጭ ዝሀ ተናጋ ጠኔ ጉያ ተግዳሮት ቁንጣን ወልቅ ናጠ ስሞታ ከጀለ መገርመም ጉራማይሌ ደልቃቃ ጽንፈኛ መተየብ \
        ህጸጽ ሟርት ሌማት ዙፋን ምስቅልና ወለል ፍርፈራ መኖ አሽከር ችሮታ ሞጨለፈ ቀለበሰ ሸለቀቀ ሸመጠጠ ዘነጠለ ቸነከረ ፈነደቀ ዘረጠጠ ወሰለተ ንትርክ ሰለቀጠ ርሚጦ ጠጠረ ተጐነጨ ተደመመ \
        ኑባሬ አሜኬላ አሽክላ አሽካካ መሳለጥ እብሪት ለዋሳ ነዛ ሁከት ለምቦጭ ሊቅ ሌጦ ልሻን እጭ ምዝበራ መዘዝ መዲና መግላሊት ምጥዋት ሙሬ ሙሾ ማርዳ ማቅ ማእቀብ ማእዘን ማግ ማጥ ምህዋር \
        ሞክሸ ዘሀ ስርወ ስውር ርዝራዥ ረግረግ ሪዝ እርቃን ሬት ሮሮ ሰነድ ሰፈነ ሲባጎ ሳዱላ ስልት ውል ባልንጀራ መሽለት ሽርደዳ ሸሸ ሸቀለ ሽንገላ ሸነሸነ ጅረት ሸንጎ ሻገተ ህብር ቀለብ መቀልወጥ ኮረዳ \
        ቅስፈት ቅራሪ ንጥር ቃረመ ቅንጣት ውጥንቅጥ ቋጥኝ ቢጤ ብላሽ ብላቴና ቡጫቂ ትልም ተመን ተረብ ቀውስ ንረት መተቸት ካነ ወከባ መዘከር እርቅ ትቢያ ትርምስ ብጣሪ ገበረ ጉቦ ግትር ገድ \
        ዘገምተኛ ጭፍራ ሽግግር ጉልላት ግሽበት መከታ ግብዝ ግኡዝ ጐሰቈለ ጥርኝ ባለሟል ባልደረባ ትካዜ ትእይንት ግብአት መነባንብ አሉባልታ አሉታ አወንታ አሳር አርጩሜ አሻራ አሽሙር አበል \
        ዳባ አቦል እክል ማነቆ እንጎቻ ንኡስ አክፍሎተኛ ሁካታ አደለ አፍላ እልፍኝ ወግ አውላላ እምቡጥ እምቅ እስስት ከሰመ ኬላ ኮረብታ ወላፈን ውቅራት መዋቅር ወንበዴ ወንደላጤ ወዘና ወጀብ \
        ወጌሻ ዘየረ መዝፈቅ ድልዝ ደምብ ድርድር ደባ ደባል ደብር ደብዛ ደንታ ደዌ ደጀን ድጎማ ደፈረሰ ዱቤ ቀመመ ዳዋ ድንክ ገባር ጎስቋላ ጠንቅ ተደለቀ መላወስ ባለጠጋ ጭላንጭል ጸኑ ጽዋ ፍልሰት \
        ፈነጠዘ ፈውስ ፍካት ፋስኮ ፋኖስ ፌዝ ፍስሀ በረበረ ትችት አበሳ አለኝታ ነጎዴ አወከ ሸርክ ትንበያ አንጓ ቀረቀረ ዋልታ ህዳግ ያሸበረቀ ሰርጥ ጉቶ መነመነ ቀዮ ወትሮ ባውዛ ዘረረ ዲቃላ ቦጭቧጫ \
        ቆጭቋጫ ሽሙጥ ሰሰነ ተፈናፈነ ጋጠ ከነከነ አንኳር ተመሳቀለ አወገዘ ገር ገታ ሰነከለ ተናጠ ግህደት ማሾር መዝገን ወረታ ገሀድ አርታኢ ገፈት አጀብ ህቡእ ቋጨ ትጉ ረገድ ዳዴ መከተ ወረሰ \
        ርህሩህ ጥልቅ ተገባደደ ወቀሳ ቀንበር ለገሰ ምሽግ አንጋዳ መናወዝ መምነሽነሽ ተንጣጣ ፋፋ ዘከረ አማ ታቀበ አገተ መነቀረ ሰረጸ አሾቀ ጥሻ ማኛ ህልውና ልቅ ለዘበ ተመረዘ አምሳል መሲና መቃ \
        መናኝ መካን ማገጠ ስልቻ ሰመረ ስስ ሱባኤ ስንዱ ስጋት አሰጋ ሾለ ሽንሽን ቂል ሻንዳ ቀላድ ቆረቆረ ቆረቆዘ ቅኝት ብሄራዊ ባልቦላ ባልቴት ብሩህ ባርኔጣ አበሰረ ቦቃ በነነ ባነነ በወዘ በዳ ቧጠጠ \
        ትሁት ታበየ ታታሪ ትእቢት ታዛ ትዝብት ተድላ ተገን መንበር አለበነበ አናወጠ ነጠበ ነጠፈ አውድ ምእላድ ምእመን እሙን እመጫት አረመኔ ወገግታ ዋቢ ዘበት ለግጥ ቦተረፈ ዘቀጠ ተዛመተ \
        ዘለሰ ዘለፈ ዘላበደ አዋዛ ውሽንፍር አውታር አሸበረቀ ዘረጦ ቀላመደ ከነበለ ወረንጦ ወመኔ ከይሲ ወሮበላ ወረጋ ከነቸረ ከነፈ ከተበ ወሰካ ወይባ ወጠነ ወገነ ወገረ ከመከመ ከረደደ ከልዋሳ ከላባ \
        ወዘፈ ተወነጨፈ ወኔ ወደመ አቆራኘ ዋለለ ሸቀሸቀ እፎይታ ከለበ ቸር ውዝፍ ለዘብተኛ ቀሳማ ትምክህት ጥቃቅን ሀተታ ሂስ ጠበብት ልሳን ሎሌ መሰዋት መስዋዕት መባ መንጋ መደዳ መድን \
        ማሳ ማእበል ሰለባ ሽመቃ ቤዛ ነበልባል አላባ አቀበት አቻ አንጋፋ አድማ አጋፋሪ አፈንጋጭ ክርታስ ክፈፍ ዋስትና ዋግ ውጥን ድንጋጌ ጥሪት ፈር ፍንጭ ሰጋ ቃኘ ተሳነ ዘነጋ ዛለ ደነገገ ተንጸባረቀ \
        ተከማቸ ተከሰተ ተዋቀረ ጀንፈል ወጋዳ ሰገባ ሰበከ ሰገሰገ ቀለመ ናሙና ትርኢት እጩ ሸረኛ እንዝላል ከረከሰ ውድማ ተዘከረ ዘከዘከ ዘውድ ዘይቤ ዛቻ ደለሰ ደቃ ደለመ ደለበ ደለዘ ድሪቶ ድሪ \
        ደሰቀ ደበተ ደቀነ ደገነ ደቀደቀ ደበለ ደባይ ደነበዘ ደነፋ ደገለለ ድንጉስ ደፈቀ ደፈጠጠ ዳሰ ደፈር ዳሸቀ ዳበረ ዳበሰ ዳተኛ ዳከረ ላቆጠ ፌስታ ድንፋታ ዶለተ አቃጣሪ ጅምላ ጀምበር ጀብድ ጃጀ \
        ገለመጠ ገለፈጠ ተገማሸረ ገረዘዘ ገረገረ አመጸ ገሰገሰ ገሰጸ ገሸለጠ ገነተረ ጎላ ገጠጠ ግፍ ግዙፍ ጎረጠ ጎረጎረ ጎሰቆለ ጎነጠ ተጎናጸፈ ጎፈነነ ጎፈየ አጓራ አጎረ ጠሀኘ ጠለለ ጠለመ ጠልሰም ተጥለቀለቀ \
        ጠለዘ ነረተ ጠረረ ጠረመሰ ጠረነቀ ጠረፈ ቀረቀበ ጠሰጠሰ ጠቢብ ጠነበሰ ጠነነ ጠነዛ ጠነጠነ ጠነፈፈ ጠና ጽናት አጽናና አጥናፍ ጠወለገ ጠገነነ ዘገነነ ጣመነ ጣረ ጦማር ጦፈ ደራ ጨመተ ጨመተረ \
        ጨመደደ ጨቦደ ጨነገዘ ጨደደ ጨፈገገ ጫተረ ጭቦ ጸና ፈለሰ ፈለቀጠ ፈለግ ፈልፈላ ፈረጅ ፈረጠጠ ፈተገ ፔዳጎጂ ተሰገሰግ ጮቤ ጨከከ ተጨናጎለ ኮርማታ ጨረፈ ጨረገደ ጧሪ አጦለ አጤነ \
        ጥጋት ጠወረ ጠነበዘ ጥንስስ ጠረቀ ጠረቃ ጓፈለ ጎደፈረ ነጎደ ጉንቁል ጎበጎበ ጎመድ ጎለመ ግድፈት ጋፈ ጋጠረ ጋንድያ ጠረንገሎ ጋመረ ጋረጠ ደነቀረ ጉግስ ጉንጉን ጉባ ኮረፕታ ጉራጅ ግግር ተጋድሎ \
        ገደደ ገዳዳ አንጋደደ ገደብ ጋየ ገናዘበ አገናዘበ ገበየ ምግባር ገሸሸ ደርባባ ገራራ ገመጠጠ ገመረ ግልግል አገለደመ ጅንግላ ዶለዝ ዳዳ አደገደገ አጎበደደ ደየነ ደወረ ደካ ደከረ ደበሰ ደረዘ ደለፈሰ \
        ተንደላቀቀ ተንዠበረረ ዝፍት ዘጠረ ዘፈቀ ዘየነ ዘነዘነ ዘርፍ ውቅር ዋጀ ዋተተ ባተለ ዋለገ ዋርዳ ወፈፈ ተወጣ ወጋገን ወደረኛ ወኪል ተወናበደ ወታፋ አዋቀረ ተወቀረ ወሸባ አመልማሎ ወሸኔ ወሰው \
        ወስዋሳ ወሰሰ ወላወለ ወላላ ወልማማ ኳሸ ተኮፈሰ ኮፈን ኮታ ኩሩ ጅንን ቁንን ኮረመተ ኩረጃ ክስተት ክሪክ ኩክኒ ጭርት አከፈለ ከፈረረ ቀፈረረ አከነፈሰ አጎበጎበ ከነተረ ክብር ልእልና ከሰተ ከረከረ \
        ከለፈ ከለበሰ ከለሰ እድር አፈገ አጀለ አድርባይ አዘቅት እኩያ አንጃ አናደለ አናጠረ አትላስ አተተ አባዜ አብነት አባሪ አበቀ አስቤዛ ማሰስ አረንዛ አመረቃ አለመጠ ነፈለለ አናጠበ ነገረፈጅ ነኮተ \
        መሸከ መንቆር ንቅስ ቶታን ቱባ ቱጃር ተአቅቦ ተነነ መልህቅ ሰሀ ስልብ ሰልካካ ስሞታ መስበቅ ሰደቃ ሰገሌ ሲናባ ሳጋ ሸምቀቆ ሽል ሻጋታ ሽንፍላ ሾተል ቀልማዳ ቅርቃብ ቅስም ቀበኛ ቀዛፊ ቀፍር\
		ቀፈት ቁር ቃሊብ ቃፊር ቅጽ ቆሽማዳ ቈርማዳ ቈንቋና ቁንጽል ቁባት ቋሊማ መቋመጥ ቋያ ቋድ ብርቅስ ቡሽ ቡቃ ባላንጣ ባልጩት ባርዳ ባና ባንኮኒ ባዘቶ ባዜቃ ባጥ ብሆር ብላሽ ብሌን ብሶል\
		ቡትቶ ቦቃ ቦና ትልም ተልካሳ ተረብ ትብትብ ትንታግ ደርባባ ጋጋታ ተገን ተላላ ታዛ ትሩፋት ችፌ ነፈለል ንፍሮ ወናፍ ናጭ ንኡስ አምባ አምቡላ አረሆ አባሪ አቦሬ አባያ እንጐቻ አንቃር አንበሪ\
		አንጋሬ አንጓ ሂስ ቡቡ ሆጃ ለምቦጭ ልግመኛ ሊጋባ ላምፋ ላይዳ ልእልና ሎፊሳ ሀኬት ሀጋይ ሁዳድ መለሎ መላላ መረን መረዋ መሲና መቅረዝ መቅን ምንዝር መግነጢስ ሙሬ ሟርት ሙጢ ሙጫ\
		ማርዳ ማቅ ማጥ ሞጋጋ ሪዝ"
        root=root.split()
        colum=["word","count"]
        lemma=dataset = pd.DataFrame(columns=colum)
        for w in root:
            lemma.loc[len(lemma.index)]=[w,0]
        lemma.to_excel(paths+'/dataset/other/complex_word.xlsx',index=False)
    #start extructing text
    column=["text","label"]
    lemma = pd.DataFrame() 
    dataset = pd.DataFrame(columns=column)
    complx = pd.DataFrame(columns=column)
    noncomplx = pd.DataFrame(columns=column)
    preproces = pd.DataFrame(columns=column)
    newsentenc=""

    #saved complex annotated data
    if os.path.exists(paths+"/dataset/dataset.xlsx"):
        complxold = pd.read_excel(paths+"/dataset/dataset.xlsx")
        os.remove(paths+"/dataset/dataset.xlsx")
        complx=pd.concat([complxold,complx]) 

    #saved reserved annotated data     
    if os.path.exists(paths+"/dataset/other/reserve.xlsx"):
        reserves = pd.read_excel(paths+"/dataset/other/reserve.xlsx")
        os.remove(paths+"/dataset/other/reserve.xlsx")
        noncomplx=pd.concat([reserves,noncomplx])

    if os.path.exists(paths+"/dataset/sentence.txt"):
        allfiles = glob.glob(paths+'/dataset/sentence.txt')#most change simple to sentence
        df = pd.concat((pd.read_csv(f, header=None, names=["text"]) for f in allfiles))
        lemma = pd.read_excel(paths+"/dataset/other/complex_word.xlsx")
        if df.empty==False:
            for sent in df["text"]:
                catch=""
                rootsent=""
                tokens=sent.split()
                for words in tokens:
                    reslt=""
                    if words not in spch:
                        wordrt=hm.anal('amh', words, um=True)
                        if wordrt!=[]:
                            wordlema=wordrt[0]['lemma'].replace("|", "/")
                            if "/" in wordlema:
                                reslt = re.search('(.*)/', wordlema)
                                reslt=reslt.group(1)
                                rootsent+=reslt+" "
                            else:
                                reslt=wordlema
                                rootsent=rootsent+" "+reslt+" "
                        else:
                            reslt=words
                            rootsent=rootsent+" "+reslt+" "
                    else:
                        reslt=words
                        rootsent=rootsent+" "+reslt+" "
                index=0
                for comp in lemma["word"].values:
                    if " "+comp+" " in rootsent:
                        catch="found"
                        if len(rootsent)<250 and len(rootsent)>10 and lemma.loc[index,'count']<50 and sent not in complx.text.values:
                            lemma.loc[index,'count']=lemma.loc[index,'count']+1
                            clear_output(wait=True)
                            print(comp)
                            print(sent)
                            newsentenc="upgraded"
                            complx.loc[len(complx.index)]=[sent,1]#[rootsent,1]
                    index+=1
                if catch=="" and len(sent)<250 and len(sent)>10 and sent not in noncomplx.text.values:
                    noncomplx.loc[len(noncomplx.index)]=[sent,0]
        if newsentenc=="":
            print("No new data found")
            
    print()
    #delete old complex terms  and save new one
    if os.path.exists(paths+"/dataset/other/complex_word.xlsx"):
        os.remove(paths+"/dataset/other/complex_word.xlsx")
        lemma.to_excel(paths+'/dataset/other/complex_word.xlsx',index=False)

    # Balance dataset size
    reserve = pd.DataFrame(columns=column)
    result=Counter(complx.label.values==1)
    comp=result[True]
    simp=result[False]
    rslt=comp-simp
    c=0
    if rslt>1:
        for i in noncomplx["text"]:
            c+=1
            if c<rslt:
                complx.loc[len(complx.index)]=[i,0]
            else:
                reserve.loc[len(reserve.index)]=[i,0]
    else:
        reserve=pd.concat([reserve,noncomplx]) 
    if os.path.exists(paths+"/dataset/sentence.txt"):
        os.remove(paths+"/dataset/sentence.txt")
    complx.to_excel(paths+'/dataset/dataset.xlsx',index=False)
    reserve.to_excel(paths+'/dataset/other/reserve.xlsx',index=False)
    print(str(len(complx))+" Sentencess are Annotated")
    print("Please find the Dataset in path: "+paths+'/dataset/dataset.xlsx')
    print()
    
    # Dataset distribution
    result=Counter(complx.label.values==1)
    comp=result[True]
    simp=result[False]
    total=comp+simp
    complx=0
    simpl=0
    if total>0:
        complx=round((comp/total)*100,1)
        simpl=round((simp/total)*100,1)
    print("data distribution: complex "+str(complx)+"%"+" Simple "+str(simpl)+"%")
    if complx>55:
          print("Data imbalancation issue please add more data to balance the distribution")
    print()
    
    #Data Pre-processing
    if newsentenc!="":
        data=pd.read_excel (paths+'/dataset/dataset.xlsx')
        pros=input("Do You Want to Pre-process the dataset Y/N")
        print("Preprocessing Data Please wait...")
        if pros=="Y" or pros=="y":
            #Remove unexpected char like \ueff
            for indexs, cell_val in enumerate(data["text"].values):
                cell_vals=cell_val.split()
                cell_val=""
                for wrd in cell_vals:
                    if wrd not in spch:
                        cell_val+=wrd+" "
                        data.loc[indexs,'text'] = cell_val

            #remove special characters         
            for indexs, cell_val in enumerate(data["text"].values):
                for i in spch:
                    cell_val=cell_val.replace(i, "")
                data.loc[indexs,'text'] = cell_val

            # remove stopwords
            for index, sentence in enumerate(data["text"].values):
                sentence=sentence.split()
                nonstop_stor=""
                for word in sentence:
                    if word not in amharicstop:
                        nonstop_stor+=word+" "
                if nonstop_stor!="":
                    data.loc[index,'text'] = nonstop_stor
            #Normalize text
            try:
                for index, sentence in enumerate(data["text"].values):
                    normalized = normalizer.normalize(sentence) 
                    data.loc[index,'text'] = normalized
            except Exception as err:
                  print()
            
            #convert to root
            try:
                for index, sent in enumerate(data["text"].values):
                    rootsent=""
                    tokens=sent.split()
                    for words in tokens:
                        reslt=""
                        if words not in spch:
                            wordrt=hm.anal('amh', words, um=True)
                            if wordrt!=[]:
                                wordlema=wordrt[0]['lemma'].replace("|", "/")
                                if "/" in wordlema:
                                    reslt = re.search('(.*)/', wordlema)
                                    reslt=reslt.group(1)
                                    rootsent+=reslt+" "
                                else:
                                    reslt=wordlema
                                    rootsent=rootsent+" "+reslt+" "
                            else:
                                reslt=words
                                rootsent=rootsent+" "+reslt+" "
                        else:
                            reslt=words
                            rootsent=rootsent+" "+reslt+" "
                    data.loc[index,'text'] = rootsent   
            except Exception as err:
                print()

            if os.path.exists(paths+"/dataset/preprocessed_data.xlsx"):
                preproces = pd.read_excel(paths+"/dataset/preprocessed_data.xlsx")
                os.remove(paths+"/dataset/preprocessed_data.xlsx")
                data=pd.concat([preproces,data])
            data.drop_duplicates()
            data.to_excel(paths+'/dataset/preprocessed_data.xlsx',index=False)
            print("Please find Preprocessed dataset in path: "+paths+'/dataset/preprocessed_data.xlsx')

        #build new vocabulary
        print()
        bvcb=input("Do You Want to Build vocab for pretrained models Y/N")
        print("Building Vocab Please wait...")
        if bvcb=="Y" or bvcb=="y":
            bildvocab()
            vocab=open(paths+"/dataset/vocab.txt",'r',encoding="utf-8")
            vocab=vocab.read()
            vocab = vocab.split()
            print()
            print("Building new vocabulary wait...")
            for sent in data["text"]:
                sent=sent.split()
                for word in sent:
                    if word not in vocab:
                        with open(paths+'/dataset/vocab.txt', 'a',encoding="utf-8") as file:
                            file.write(word+"\n")
                        vocab=open(paths+"/dataset/vocab.txt",'r',encoding="utf-8")
                        vocab=vocab.read()
                        vocab = vocab.split()
            print()
            print("Total vocabulary built: "+str(len(vocab)))
            print("Please find vocabularies in path: "+paths+'/dataset/vocab.txt')

def bildvocab():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    path = os.getcwd()
    paths=""
    if path=="/content":
        paths="/content/drive/MyDrive/"
    else:
        paths=path
    if not os.path.exists(paths+"/dataset/vocab.txt"):
        vocab="[PAD] [unused0] [unused1] [unused2] [unused3] [unused4] [unused5] [unused6] [unused7] [unused8] [unused9] [unused10] [unused11] [unused12]\
        [unused13] [unused14] [unused15] [unused16] [unused17] [unused18] [unused19] [unused20] [unused21] [unused22] [unused23] [unused24] [unused25] \
        [unused26] [unused27] [unused28] [unused29] [unused30] [unused31] [unused32] [unused33] [unused34] [unused35] [unused36] [unused37] [unused38] \
        [unused39] [unused40] [unused41] [unused42] [unused43] [unused44] [unused45] [unused46] [unused47] [unused48] [unused49] [unused50] [unused51] \
        [unused52] [unused53] [unused54] [unused55] [unused56] [unused57] [unused58] [unused59] [unused60] [unused61] [unused62] [unused63] [unused64] \
        [unused65] [unused66] [unused67] [unused68] [unused69] [unused70] [unused71] [unused72] [unused73] [unused74] [unused75] [unused76] [unused77] \
        [unused78] [unused79] [unused80] [unused81] [unused82] [unused83] [unused84] [unused85] [unused86] [unused87] [unused88] [unused89] [unused90] \
        [unused91] [unused92] [unused93] [unused94] [unused95] [unused96] [unused97] [unused98] [UNK] [CLS] [SEP] [MASK] [unused99] [unused100] [unused101] \
        [unused102] [unused103] [unused104] [unused105] [unused106] [unused107] [unused108] [unused109] [unused110] [unused111] [unused112] [unused113] \
        [unused114] [unused115] [unused116] [unused117] [unused118] [unused119] [unused120] [unused121] [unused122] [unused123] [unused124] [unused125] \
        [unused126] [unused127] [unused128] [unused129] [unused130] [unused131] [unused132] [unused133] [unused134] [unused135] [unused136] [unused137] \
        [unused138] [unused139] [unused140] [unused141] [unused142] [unused143] [unused144] [unused145] [unused146] [unused147] [unused148] [unused149] \
        [unused150] [unused151] [unused152] [unused153] [unused154] [unused155] [unused156] [unused157] [unused158] [unused159] [unused160] [unused161] \
        [unused162] [unused163] [unused164] [unused165] [unused166] [unused167] [unused168] [unused169] [unused170] [unused171] [unused172] [unused173] \
        [unused174] [unused175] [unused176] [unused177] [unused178] [unused179] [unused180] [unused181] [unused182] [unused183] [unused184] [unused185] \
        [unused186] [unused187] [unused188] [unused189] [unused190] [unused191] [unused192] [unused193] [unused194] [unused195] [unused196] [unused197] \
        [unused198] [unused199] [unused200] [unused201] [unused202] [unused203] [unused204] [unused205] [unused206] [unused207] [unused208] [unused209] \
        [unused210] [unused211] [unused212] [unused213] [unused214] [unused215] [unused216] [unused217] [unused218] [unused219] [unused220] [unused221] \
        [unused222] [unused223] [unused224] [unused225] [unused226] [unused227] [unused228] [unused229] [unused230] [unused231] [unused232] [unused233] \
        [unused234] [unused235] [unused236] [unused237] [unused238] [unused239] [unused240] [unused241] [unused242] [unused243] [unused244] [unused245] \
        [unused246] [unused247] [unused248] [unused249] [unused250] [unused251] [unused252] [unused253] [unused254] [unused255] [unused256] [unused257] \
        [unused258] [unused259] [unused260] [unused261] [unused262] [unused263] [unused264] [unused265] [unused266] [unused267] [unused268] [unused269] \
        [unused270] [unused271] [unused272] [unused273] [unused274] [unused275] [unused276] [unused277] [unused278] [unused279] [unused280] [unused281] \
        [unused282] [unused283] [unused284] [unused285] [unused286] [unused287] [unused288] [unused289] [unused290] [unused291] [unused292] [unused293] \
        [unused294] [unused295] [unused296] [unused297] [unused298] [unused299] [unused300] [unused301] [unused302] [unused303] [unused304] [unused305] \
        [unused306] [unused307] [unused308] [unused309] [unused310] [unused311] [unused312] [unused313] [unused314] [unused315] [unused316] [unused317] \
        [unused318] [unused319] [unused320] [unused321] [unused322] [unused323] [unused324] [unused325] [unused326] [unused327] [unused328] [unused329] \
        [unused330] [unused331] [unused332] [unused333] [unused334] [unused335] [unused336] [unused337] [unused338] [unused339] [unused340] [unused341] \
        [unused342] [unused343] [unused344] [unused345] [unused346] [unused347] [unused348] [unused349] [unused350] [unused351] [unused352] [unused353] \
        [unused354] [unused355] [unused356] [unused357] [unused358] [unused359] [unused360] [unused361] [unused362] [unused363] [unused364] [unused365] \
        [unused366] [unused367] [unused368] [unused369] [unused370] [unused371] [unused372] [unused373] [unused374] [unused375] [unused376] [unused377] \
        [unused378] [unused379] [unused380] [unused381] [unused382] [unused383] [unused384] [unused385] [unused386] [unused387] [unused388] [unused389] \
        [unused390] [unused391] [unused392] [unused393] [unused394] [unused395] [unused396] [unused397] [unused398] [unused399] [unused400] [unused401] \
        [unused402] [unused403] [unused404] [unused405] [unused406] [unused407] [unused408] [unused409] [unused410] [unused411] [unused412] [unused413] \
        [unused414] [unused415] [unused416] [unused417] [unused418] [unused419] [unused420] [unused421] [unused422] [unused423] [unused424] [unused425] \
        [unused426] [unused427] [unused428] [unused429] [unused430] [unused431] [unused432] [unused433] [unused434] [unused435] [unused436] [unused437] \
        [unused438] [unused439] [unused440] [unused441] [unused442] [unused443] [unused444] [unused445] [unused446] [unused447] [unused448] [unused449] \
        [unused450] [unused451] [unused452] [unused453] [unused454] [unused455] [unused456] [unused457] [unused458] [unused459] [unused460] [unused461] \
        [unused462] [unused463] [unused464] [unused465] [unused466] [unused467] [unused468] [unused469] [unused470] [unused471] [unused472] [unused473] \
        [unused474] [unused475] [unused476] [unused477] [unused478] [unused479] [unused480] [unused481] [unused482] [unused483] [unused484] [unused485] \
        [unused486] [unused487] [unused488] [unused489] [unused490] [unused491] [unused492] [unused493] [unused494] [unused495] [unused496] [unused497] \
        [unused498] [unused499] [unused500] [unused501] [unused502] [unused503] [unused504] [unused505] [unused506] [unused507] [unused508] [unused509] \
        [unused510] [unused511] [unused512] [unused513] [unused514] [unused515] [unused516] [unused517] [unused518] [unused519] [unused520] [unused521] \
        [unused522] [unused523] [unused524] [unused525] [unused526] [unused527] [unused528] [unused529] [unused530] [unused531] [unused532] [unused533] \
        [unused534] [unused535] [unused536] [unused537] [unused538] [unused539] [unused540] [unused541] [unused542] [unused543] [unused544] [unused545] \
        [unused546] [unused547] [unused548] [unused549] [unused550] [unused551] [unused552] [unused553] [unused554] [unused555] [unused556] [unused557] \
        [unused558] [unused559] [unused560] [unused561] [unused562] [unused563] [unused564] [unused565] [unused566] [unused567] [unused568] [unused569] \
        [unused570] [unused571] [unused572] [unused573] [unused574] [unused575] [unused576] [unused577] [unused578] [unused579] [unused580] [unused581] \
        [unused582] [unused583] [unused584] [unused585] [unused586] [unused587] [unused588] [unused589] [unused590] [unused591] [unused592] [unused593] \
        [unused594] [unused595] [unused596] [unused597] [unused598] [unused599] [unused600] [unused601] [unused602] [unused603] [unused604] [unused605] \
        [unused606] [unused607] [unused608] [unused609] [unused610] [unused611] [unused612] [unused613] [unused614] [unused615] [unused616] [unused617] \
        [unused618] [unused619] [unused620] [unused621] [unused622] [unused623] [unused624] [unused625] [unused626] [unused627] [unused628] [unused629] \
        [unused630] [unused631] [unused632] [unused633] [unused634] [unused635] [unused636] [unused637] [unused638] [unused639] [unused640] [unused641] \
        [unused642] [unused643] [unused644] [unused645] [unused646] [unused647] [unused648] [unused649] [unused650] [unused651] [unused652] [unused653] \
        [unused654] [unused655] [unused656] [unused657] [unused658] [unused659] [unused660] [unused661] [unused662] [unused663] [unused664] [unused665] \
        [unused666] [unused667] [unused668] [unused669] [unused670] [unused671] [unused672] [unused673] [unused674] [unused675] [unused676] [unused677] \
        [unused678] [unused679] [unused680] [unused681] [unused682] [unused683] [unused684] [unused685] [unused686] [unused687] [unused688] [unused689] \
        [unused690] [unused691] [unused692] [unused693] [unused694] [unused695] [unused696] [unused697] [unused698] [unused699] [unused700] [unused701] \
        [unused702] [unused703] [unused704] [unused705] [unused706] [unused707] [unused708] [unused709] [unused710] [unused711] [unused712] [unused713] \
        [unused714] [unused715] [unused716] [unused717] [unused718] [unused719] [unused720] [unused721] [unused722] [unused723] [unused724] [unused725] \
        [unused726] [unused727] [unused728] [unused729] [unused730] [unused731] [unused732] [unused733] [unused734] [unused735] [unused736] [unused737] \
        [unused738] [unused739] [unused740] [unused741] [unused742] [unused743] [unused744] [unused745] [unused746] [unused747] [unused748] [unused749] \
        [unused750] [unused751] [unused752] [unused753] [unused754] [unused755] [unused756] [unused757] [unused758] [unused759] [unused760] [unused761] \
        [unused762] [unused763] [unused764] [unused765] [unused766] [unused767] [unused768] [unused769] [unused770] [unused771] [unused772] [unused773] \
        [unused774] [unused775] [unused776] [unused777] [unused778] [unused779] [unused780] [unused781] [unused782] [unused783] [unused784] [unused785] \
        [unused786] [unused787] [unused788] [unused789] [unused790] [unused791] [unused792] [unused793] [unused794] [unused795] [unused796] [unused797] \
        [unused798] [unused799] [unused800] [unused801] [unused802] [unused803] [unused804] [unused805] [unused806] [unused807] [unused808] [unused809] \
        [unused810] [unused811] [unused812] [unused813] [unused814] [unused815] [unused816] [unused817] [unused818] [unused819] [unused820] [unused821] \
        [unused822] [unused823] [unused824] [unused825] [unused826] [unused827] [unused828] [unused829] [unused830] [unused831] [unused832] [unused833] \
        [unused834] [unused835] [unused836] [unused837] [unused838] [unused839] [unused840] [unused841] [unused842] [unused843] [unused844] [unused845] \
        [unused846] [unused847] [unused848] [unused849] [unused850] [unused851] [unused852] [unused853] [unused854] [unused855] [unused856] [unused857] \
        [unused858] [unused859] [unused860] [unused861] [unused862] [unused863] [unused864] [unused865] [unused866] [unused867] [unused868] [unused869] \
        [unused870] [unused871] [unused872] [unused873] [unused874] [unused875] [unused876] [unused877] [unused878] [unused879] [unused880] [unused881] \
        [unused882] [unused883] [unused884] [unused885] [unused886] [unused887] [unused888] [unused889] [unused890] [unused891] [unused892] [unused893] \
        [unused894] [unused895] [unused896] [unused897] [unused898] [unused899] [unused900] [unused901] [unused902] [unused903] [unused904] [unused905] \
        [unused906] [unused907] [unused908] [unused909] [unused910] [unused911] [unused912] [unused913] [unused914] [unused915] [unused916] [unused917] \
        [unused918] [unused919] [unused920] [unused921] [unused922] [unused923] [unused924] [unused925] [unused926] [unused927] [unused928] [unused929] \
        [unused930] [unused931] [unused932] [unused933] [unused934] [unused935] [unused936] [unused937] [unused938] [unused939] [unused940] [unused941] \
        [unused942] [unused943] [unused944] [unused945] [unused946] [unused947] [unused948] [unused949] [unused950] [unused951] [unused952] [unused953] \
        [unused954] [unused955] [unused956] [unused957] [unused958] [unused959] [unused960] [unused961] [unused962] [unused963] [unused964] [unused965] \
        [unused966] [unused967] [unused968] [unused969] [unused970] [unused971] [unused972] [unused973] [unused974] [unused975] [unused976] [unused977] \
        [unused978] [unused979] [unused980] [unused981] [unused982] [unused983] [unused984] [unused985] [unused986] [unused987] [unused988] [unused989] \
        [unused990] [unused991] [unused992] [unused993]"
        vocab = vocab.split()
        with open(paths+'/dataset/vocab.txt', 'a',encoding="utf-8") as file:
            for i in vocab:
                file.write(i+"\n")
                
def amTextPreprocessing(textss):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    path = os.getcwd()
    paths=""
    if path=="/content":
        paths="/content/drive/MyDrive/"
    else:
        paths=path
    if not os.path.exists(paths+"/dataset"):
        os.makedirs(paths+"/dataset")
    if not os.path.exists(paths+"/dataset/stoplist"):
        os.makedirs(paths+"/dataset/stoplist") 

    if not os.path.exists(paths+"dataset/stoplist/spchar.txt"):
        spch="፩ ፫ ፪ ፬ ፭ ፮ ፯ ፰ ፱ ፲ ፳ ፴ ፵ ፶ ፷ ፸ ፹ ፶ ፺ ፯ ፻ ፼ 0 1 2 3 4 5 6 7 8 9 { } a A b B c C d D e E f F g G h H i I j J k K l L m M \
        n N o O p P q Q r R s S t T u U v V w W x X y Y z Z '"' | :\ ; , . / < > ? [ ] ; , . / ፤ ፣ ። ፡  ” “ ፠ ፥ ፦ ፧ ፨ \ ´ … \
        !「 "'" ¦ _ , \ ¨ ፣ ፤ . ፹ ። ~ ! @ # $ % ^ & * ( ) _ + ` ፟ = - – \ufeff • ★ 🙂 � "
        spch = spch.split()
        with open(paths+'/dataset/stoplist/spchar.txt', 'a',encoding="utf-8") as file:
            for i in spch:
                file.write(i+"\n")

    if not os.path.exists(paths+"/dataset/stoplist/amharic_stop_lists.txt"):
        amstop="የ ለ በዚህ እንደ ነገር አንድ አንድን እና አለ አየ የት ግኝ በላ ሆነ ለየ ባለ ጊዜ ሄደ በ አል ሃ ያ ጋ ሆነ ነገረ ነበረ ወይም ሆኑ ሆኖም ነው ናቸው ነበር ሁሉንም ላይ ሌላ ሌሎች ስለ \
        ቢሆን ብቻ መሆኑ ማለት ማለቱ የሚገኝ የሚገኙ ማድረግ ማን ማንም ሲሆን ሲል እዚህ እንጂ በኩል በውስጥ በጣም ይህን በተለይ እያንዳንድ በሆነ ከዚህ ከላይ ከመሀል ከመካከል ከጋራ ጋራ ወዘተ \
        ወደ ያለ ሲሉ በተመለከተ በተመሳሳይ ያሉ የኋላ የሰሞኑ  ሁሉ ሁሉም ኋላ ሁኔታ ሆነ ሆኑ ሆኖም ሁል ሁሉንም ላይ ሌላ ሌሎች ልዩ መሆኑ ማለት ማለቱ መካከል የሚገኙ የሚገኝ ማድረግ ማን \
        ማንም ሰሞኑን ሲሆን ሲል ሲሉ ስለ ቢቢሲ ቢሆን ብለዋል ብቻ ብዛት ብዙ ቦታ በርካታ በሰሞኑ በታች በኋላ እባክህ በኩል በውስጥ በጣም ብቻ በተለይ በተመለከተ በተመሳሳይ የተለያየ የተለያዩ \
        ተባለ ተገለጸ ተገልጿል ተጨማሪ ተከናውኗል ችግር ታች ትናንት ነበረች ነበሩ ነበረ ነው ነይ ነገር ነገሮች ናት ናቸው አሁን አለ አስታወቀ አስታውቀዋል አስታውሰዋል እስካሁን አሳሰበ አሳስበዋል \
        አስፈላጊ አስገነዘቡ አስገንዝበዋል አብራርተዋል እባክዎ አንድ አንጻር እስኪደርስ እንኳ እስከ እዚሁ እና እንደ እንደገለጹት እንደተገለጸው እንደተናገሩት እንደአስረዱት እንደገና ወቅት እንዲሁም \
        እንጂ እዚህ እዚያ እያንዳንዱ እያንዳንዳችው እያንዳንዷ ከ ከኋላ ከላይ ከመካከል ከሰሞኑ ከታች ከውስጥ ከጋራ ከፊት ወዘተ ወይም ወደ ወደፊት ውስጥ እባክሸ ውጪ ያለ ያሉ ይገባል የኋላ የሰሞኑ \
        የታች የውስጥ የጋራ ያ ይታወሳል ይህ ደግሞ ድረስ ጋራ ግን ገልጿል ገልጸዋል ግዜ ጥቂት ፊት ደግሞ ዛሬ ጋር ተናግረዋል የገለጹት ይገልጻል ሲሉ ብለዋል ስለሆነ አቶ ሆኖም መግለጹን አመልክተዋል \
        ይናገራሉ አበራርተው አስረድተዋል እስከ ይህ ከነ ያለ ወደ ስለ ተራ ሙሉ ጋር እና ነው ግን ወይም እንጅ እንኳ ናቸው አዎን እንዲህ እነዚህ ምን ይኸውም"
        amstop = amstop.split()
        with open(paths+'/dataset/stoplist/amharic_stop_lists.txt', 'a',encoding="utf-8") as file:
            for i in amstop:
                file.write(i+"\n")
    spch=open(paths+"/dataset/stoplist/spchar.txt",'r',encoding="utf-8").read().split()
    amharicstop=open(paths+"/dataset/stoplist/amharic_stop_lists.txt",'r',encoding="utf-8").read().split()
    
    colum=["text"]
    data = pd.DataFrame(columns=colum)
    sentences=re.split('[?።!\n]', textss)
    print("Preprocessing Data Please wait...")
    for i in sentences:
        data.loc[len(data.index)]=[i]
    #Remove unexpected char like \ueff
    for indexs, cell_val in enumerate(data["text"].values):
        cell_vals=cell_val.split()
        cell_val=""
        for wrd in cell_vals:
            if wrd not in spch:
                cell_val+=wrd+" "
                data.loc[indexs,'text'] = cell_val

    #remove special characters         
    for indexs, cell_val in enumerate(data["text"].values):
        for i in spch:
            cell_val=cell_val.replace(i, "")
        data.loc[indexs,'text'] = cell_val

    # remove stopwords
    for index, sentence in enumerate(data["text"].values):
        sentence=sentence.split()
        nonstop_stor=""
        for word in sentence:
            if word not in amharicstop:
                nonstop_stor+=word+" "
        if nonstop_stor!="":
            data.loc[index,'text'] = nonstop_stor
    #Normalize text
    try:
        for index, sentence in enumerate(data["text"].values):
            normalized = normalizer.normalize(sentence) 
            data.loc[index,'text'] = normalized
    except Exception as err:
          print()
    #convert to root
    try:
        for index, sent in enumerate(data["text"].values):
            rootsent=""
            tokens=sent.split()
            for words in tokens:
                reslt=""
                if words not in spch:
                    wordrt=hm.anal('amh', words, um=True)
                    if wordrt!=[]:
                        wordlema=wordrt[0]['lemma'].replace("|", "/")
                        if "/" in wordlema:
                            reslt = re.search('(.*)/', wordlema)
                            reslt=reslt.group(1)
                            rootsent+=reslt+" "
                        else:
                            reslt=wordlema
                            rootsent=rootsent+" "+reslt+" "
                    else:
                        reslt=words
                        rootsent=rootsent+" "+reslt+" "
                else:
                    reslt=words
                    rootsent=rootsent+" "+reslt+" "
            data.loc[index,'text'] = rootsent   
    except Exception as err:
        print()

    for sent in data["text"]:
          if sent !="":
            with open(paths+'/dataset/preprocessed_data.txt', 'a',encoding="utf-8") as file:
                file.write(sent+"\n")
    print("please find the preprocesed dataset in path: "+paths+'dataset/preprocessed_data.txt')