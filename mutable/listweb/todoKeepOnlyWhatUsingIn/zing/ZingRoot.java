package mutable.listweb.todoKeepOnlyWhatUsingIn.zing;
import static mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.CommonFuncs.lg;

import java.security.DigestException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.*;

import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Err;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.err.Todo;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.designsImConsidering.SimpleZingVM;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.designsImConsidering.ZingVM;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl.LongArrayZing;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.impl.SimpleVar;
import mutable.listweb.todoKeepOnlyWhatUsingIn.zing.langs.StringLang;
import immutable.occamsjsonds.JsonDS;

/** root namespace and util class.
namespace, where both name and object are zings.
TODO garbcol manually or by weakref? If by weakref, then must keep those zings as separate objects in memory,
as a real tree even though a nonleaf branch is also a bitstring zing.
*/
public class ZingRoot{
	private ZingRoot(){}
	
	public static final String n = JsonDS.n;
	
	static{
		System.err.println("TODO redesignZingToHaveBasicTypes (and that may be replaced by or become an optimized subtype of ufnode/funode later), in benfrayfields mindmap theres 10 bits for header:"
			+n+"This will probably be a different order, but just for example:"
			+n+"isMap"
			+n+"isList"
			+n+"isNDimArray //of bits, so each data must be the same size, and core type of that data is told by rest of header"
			+n+"isHash"
			+n+"isScalars"
			+n+"isIntegers"
			+n+"isStrings //Example codeString or normalString"
			+n+"isFuncs //Example codeString or maplist which happens to define a func"
			+n+"isDirty //any use of \"javaName:...\" instead of \"javaCode:...\", or anything that can change after its hashed."
			+n+"isBig"
			+n+"bits: listSize/numberOfDims (this ends at isBig?64:16)"
			+n+"bits: totalSizeInBits"
			+n+"(if isMap, isList, or isNDimArray) listSize/numberOfDims integers (which are each 16 or 64 bits)."
			+n+"content //If isHash, starts with ipfsHashAlgorithmPrefix"
		);
	}
	
	public static final long maskIsBig = 1L<<63;
	
	public static final long maskIsMap = 1L<<62;
	
	public static final long maskIsList = 1L<<61;
	
	public static final long maskIsHash = 1L<<60;
	
	
	
	/** the root VM which starts with all the memoryMoney and computeMoney available now or later in this JVM *
	public static final ZingVM vm = new SimpleZingVM(Long.MAX_VALUE, Long.MAX_VALUE);
	*/
	
	static{ lg("TODO in Zing interface: java stream api, instead of just iterators? Spliterator. Stream. DoubleStream, etc."); }
	
	/** all sizes are in bits. This is the max size to include a child's bits directly
	in parent instead of a hash of the child.
	This happens to the same size as width of a Linux terminal, 80 bytes,
	but was chosen for what kind of hashes and type datastructs can fit in it.
	<br><br>
	Its important this be small since child literals and child hashes are often map keys
	which are sorted by comparing them as bitstrings, first by length then by bits,
	and if the lengths equal its a worst case of linear search of child bit size,
	especially when childs are sorted near eachother in the end of a binarySearch
	or in a linearSearch of a small map.
	<br><br>
	Its important this be big enough to hold the kinds of secureHash likely
	to be used in this software, including sha2 and sha3 up to 512 bits, by default 256 bits.
	<br><br>
	Its important this be big enough to hold a literal list of 2 256-bit hashes,
	(which is the default hash size) since the type system is those size 2 lists
	at certain places alternating type value type value in recursion,
	and bitstring leaf is optionally preceded by a type. That way, type can be cached,
	and no jumping outside the map in memory to loop or binarySearch.
	Its ok the bigger kinds of hashes will have to store list(type,value)
	as hash instead of literal if type is a hash. Theres still a simpler kind of type
	that fits there, where type is a bitstring
	such as arbitrary text 'txt' or 'treemap' or 'treelist'.
	Since every !isBig() header is 32 bits (else 128),
	and every 256 bit hash (including header) is 296 bits,
	a list of 2 such hashes is 624 bits. The next multiple of 64 bits is 640,
	so I chose that constant.
	*/
	public static final int maxChildLiteralSize = 640;
	
	/** TODO zingLazyHashStub for weakReferences */ 
	private static WeakHashMap<Zing,Zing> hashToOb = new WeakHashMap();
	
	//private static Zing state = "todo try load state from file, else create empty state";
	
	public static synchronized Zing get(Zing hashname){
		return hashToOb.get(hashname);
	}
	
	/** FIXME How can this work with LongArrayZing.hash(byte hashAlg) if always uses defaultHashAlg? */ 
	public static synchronized void put(Zing z){
		if(z.isHash()) throw new Err("already a hash");
		hashToOb.put(z.hash(defaultHashAlg), z);
	}
	
	/** returns sorted array of all hashes you can get(hash) */
	public static synchronized Zing[] hashes(){
		Zing h[] = hashToOb.keySet().toArray(new Zing[0]);
		Arrays.sort(h);
		return h;
	}
	
	/*
	"TODO what should TestZingRbmTreeUi call here, and how should it get the ZingVideo, listweb ui, jsoundcard zing, etc?"
	+" How should it choose to have which set of those or multiple of each? Maybe there should be only 1 of each,"
	+" listwebui on left, ZingVideo (including keyboard and mouse input) most of the size on the right,"
	+" jsoundcard on speakers and microphone. All of these are optional. Each should have a name and put"
	+" what to do with it in a key in the map, maybe mapped from 'ui' key to that map of specific kinds of ui."
	+" However big the video is, maybe stretch that by the existing magnify ability of ZingVideo, or do I want scrolling and rotation which it also can do?"
	*/
	
	/** The whole system is reachable from a forest root per thread.
	TODO The main thread's zing (its state) is saved on harddrive, along with everything reachable from it.
	*/
	public static ThreadLocal<Var<Zing>> threadState = new ThreadLocal<Var<Zing>>(){
		protected Var<Zing> initialValue(){ return new SimpleVar(ZingRoot.emptyMap); }
	};
	
	/** root state per computer.
	Calling setState makes everything not reachable from it available for garbcol,
	so make sure to include everything you want from prev state, reachable on some path.
	*
	public static synchronized void setState(Zing z){
		state = z;
	}
	
	/** root state per computer *
	public static synchronized Zing getState(){
		return state;
	}*/
	//"TODO use a ThreadLocal of a Var<Zing> since each thread should have its own state, and one of them should be the main thread, and all of them should be named threads maybe hooking into listweb for those names"
	
	public static final Lang<String> langString = new StringLang();
	
	/** convert Zing of text utf8 to string */
	public static final String s(Zing textUtf8){
		return langString.o(textUtf8);
	}

	/** convert string to zing of text utf8 */
	public static final Zing s(String text){
		return langString.z(text);
	}
	
	public static final boolean ptrWouldHash(Zing z){
		//UPDATE: a hash must never be bigger than maxChildLiteralSize
		//if(z.isHash()) return false; //FIXME limit hash size to that? its technically possible but unlikely for a hash to be bigger than maxChildLiteralSize
		if(maxChildLiteralSize < z.sizeBits()) return true;
		long sizeList = z.sizeList();
		if(2 < sizeList) return true;
		if(0 < sizeList && z.listGet(0).isMaplist()) return true;
		if(1 < sizeList && z.listGet(1).isMaplist()) return true;
		return false;
	}
	
	/** Pointer, how to include a zing in another zing and stay small.
	If z is small enough (TODO what standard size limit), returns z,
	else returns z.hash(standard hash alg).
	*/
	public static Zing ptr(Zing z, byte hashAlg){
		return ptrWouldHash(z) ? z.hash(hashAlg) : z;
		//return z.sizeBits()<=maxChildLiteralSize ? z : z.hash(hashAlg);
		//int maxLiteralSize = Hash.size(hashAlg)*2; //wont work because multiple available hash sizes
	}
	
	/** uses Hash.defaultHashAlg */
	public static Zing ptr(Zing z){
		//return ptr(z,defaultHashAlg);
		return ptrWouldHash(z) ? z.hash(defaultHashAlg) : z;
	}
	
	public static UnaryOperator<Zing> hasher(byte hashAlg){
		UnaryOperator<Zing> z = hasher[hashAlg&0xff];
		if(z == null){
			final String algName = ByteToName.get(hashAlg);
			z = hasher[hashAlg&0xff] = (Zing zz)->{
				long zzSize = zz.sizeBits();
				if((zzSize&7) != 0) throw new Todo(
					"bit alignment. Example: sha256 is well defined for sizes not a multiple of 8 bits, and I have an implementation of it somewhere that can be slightly modified to do bit alignment (it has a check for if its a multiple of 8 and throws if not). size="+zzSize);
				try{
					MessageDigest md = MessageDigest.getInstance(algName);
					/*if((cSize>>>3) <= Integer.MAX_VALUE){ //use zing.ba(int) func for efficiency
						int byteSize = (int)(cSize>>>3);
						for(int b=0; b<byteSize; b++){
							md.update(content.ba(b));
						}
					}else{ //read bit aligned, which happens to be byte aligned but need the long range
						for(long i=0; i<cSize; i+=8){
							md.update(content.b(i));
						}
					}*/
					long end = zzSize>>>3;
					for(long i=0; i<end; i++){
						md.update(zz.ba(i));
					}
					int innerByteSize = md.getDigestLength();
					byte wrappedHash[] = new byte[5+innerByteSize];
					//isBig bit is 0
					//isMap bit is 0
					//isList bit is 0
					//isHash bit is 1
					//then uint12 listSize is 0
					//then uint16 size of returned zing in bits is wrappedHash.length*8
					//then hashAlg byte
					//then hash
					wrappedHash[0] = (byte)0x10;
					wrappedHash[1] = (byte)0;
					int retSize = wrappedHash.length*8;
					wrappedHash[2] = (byte)(retSize>>>8);
					wrappedHash[3] = (byte)retSize;
					wrappedHash[4] = hashAlg;
					md.digest(wrappedHash, 5, innerByteSize);
					return new LongArrayZing(wrappedHash);
				}catch(NoSuchAlgorithmException | DigestException e){
					throw new Err(e);
				}
			};
		}
		return z;
	}

	//"TODO IMPORTANT ANSWER ASAP todo is ipfs hashAlg a byte or Byte? Its 16 bits of something, but is that a byte for hashAlg and a byte for size?"
	//"http://ethereum.stackexchange.com/questions/6861/what-datatype-should-i-use-for-an-ipfs-address-hash says its a byte for hashAlg and a byte for size"
	

	/** Same as IPFS multihash, for example one copy of it is at
	https://github.com/multiformats/multihash#table-for-multihash-v100-rc-semver
	which was linked in https://www.reddit.com/r/crypto/comments/5etmgb/are_there_smaller_standard_names_for_hash/
	but make sure to check multiple versions of it (TODO I havent done that yet) before
	accepting changes. Only accept what the world agrees on as IPFS is opensource.
	<br><br>
	https://github.com/jbenet/random-ideas/issues/1 Y2016M11 says
	"Ideally, for proper future proofing, we want a varint. Though it is to be noted
	that varints are annoying to parse + slower than fixed-width ints. There are so
	few "widely used...hash functions" that it may be okay to get away with one byte.
	Luckily, can wait until we reach 127 functions before we have to decide which one :)"
	<br><br>
	IPFS hash algorithm abbrevs are 16 bits.
	*/
	public static final Map<String,Byte> nameToByte;
	public static final Map<Byte,String> ByteToName;
	private static final UnaryOperator<Zing> hasher[];
	static{
		hasher = new UnaryOperator[256];
		Map<String,Byte> n2i = new HashMap();
		//n2i.put("identity", (byte)0x00);
		n2i.put("SHA1", (byte)0x11);
		n2i.put("SHA-256", (byte)0x12);
		n2i.put("SHA-512", (byte)0x13);
		n2i.put("SHA3-512", (byte)0x14);
		n2i.put("SHA3-384", (byte)0x15);
		n2i.put("SHA3-256", (byte)0x16);
		n2i.put("SHA3-224", (byte)0x17);
		
		//https://github.com/multiformats/multihash#table-for-multihash-v100-rc-semver says
		//"0x0400-0x040f reserved for application specific functions"
		//reserved for ipfs opcodes, probably wont ever be used in this software
		//Which of these is it?
		//n2i.put("ipfsOpcode", (byte)0x40);
		//n2i.put("ipfsOpcode", (byte) 0x00 to 0x0f);
		
		/*
		n2i.put("shake-128", (byte)0x18);
		n2i.put("shake-256", (byte)0x19);
		"TODO whats the overlap between sha3, keccak, and shake? I dont want to duplicate constants in the byte hashAlg if not enough others are using those consts, maybe resulting from a misunderstanding of what equals what else. Wikipedia somewhere says sha3 uses keccak of twice the size for its internal calculations, or something like that.."
		n2i.put("keccak-224", (byte)0x1a);
		n2i.put("keccac-256", (byte)0x1b);
		n2i.put("keccak-384", (byte)0x1c);
		n2i.put("kekkac-512", (byte)0x1d);
		*/
		
		n2i.put("blake2b", (byte)0x40);
		n2i.put("blake2s", (byte)0x41);
		nameToByte = Collections.unmodifiableMap(n2i);
		Map<Byte,String> i2n = new HashMap();
		for(Map.Entry<String,Byte> entry : nameToByte.entrySet()){
			i2n.put(entry.getValue(), entry.getKey());
		}
		ByteToName = Collections.unmodifiableMap(i2n);
		/*exports.defaultLengths = {
		  0x11: 20,
		  0x12: 32,
		  0x13: 64,
		  0x17: 28,
		  0x16: 32,
		  0x15: 48,
		  0x14: 64,
		  0x18: 32,
		  0x19: 64,
		  0x1A: 28,
		  0x1B: 32,
		  0x1C: 48,
		  0x1D: 64,
		  0x40: 64,
		  0x41: 32
		}*/
	}

	/** Dont change this unless you want to force a fork in a potentially global network.
	Multiple hashAlg can be used together without changing this,
	by using aZing.hash(byte) before the first call of aZing.hash()
	or by always using aZing.hash(byte).
	TODO should this be sha3-256 instead of sha256 (sha2-256)?
	I can only do that if there are opensource both java and javascript
	implementations of sha3-256 and its fast enough and its believed by unbiased experts
	to be harder to create any 2 bitstrings that collide,
	and it must be able to hash bitstrings of size not a multiple of 8 bits.
	<br><br>
	https://www.reddit.com/r/crypto/comments/5fy0so/what_256_bit_hash_algorithm_is_strongest_against/
	http://stackoverflow.com/questions/14356526/whats-the-difference-between-the-hash-algorithms-sha-2-and-sha-3
	QUOTE
	However, since then, the feared attacks on SHA-2 have failed to materialize, and it's now generally
	accepted that breaking SHA-2 won't be as easy as it seemed five years ago. Thus, all the variants of
	SHA-2 are still considered secure for the foreseeable future. However, since NIST had promised
	that SHA-3 would be chosen in 2012, and since a lot of people had spent quite a bit of time and
	effort on submitting and evaluating new hash functions for the competition, and since there were
	some really nice designs among the finalists, it would've seemed a shame not to choose any of them
	as the winner after all. So NIST decided to select Keccak as SHA-3, and to recommend it as an
	alternative (not successor) to the SHA-2 hash functions.
	UNQUOTE
	QUOTE
	one of the stated reasons why NIST chose Keccak over the other SHA-3 competition finalists was its
	dissimilarity to the existing SHA-1/2 algorithms; it was argued that this dissimilarity makes it
	a better complement to the existing SHA-2 algorithms (which are still considered secure and
	recommended by NIST), as well as making it less likely that any future cryptanalytic breakthroughs
	would compromise the security of both SHA-2 and SHA-3.
	UNQUOTE.
	What about other algorithms?
	*/
	public static final Byte defaultHashAlg = nameToByte.get("SHA-256");
	static{ lg("Consider SHA3-256 for defaultHashAlg before this software spreads. Probably wont choose that because its said sha2 is better optimized for running on cpus, and I'm not seeing much opensource implementations of sha3 such as gnu.crypto has sha2 but not sha3, which maybe means sha3 has efficiency problems being coded without GPU ???"); }
	
	/** all sizes are in bits */
	public static short hashSize(byte hashAlg){
		throw new Todo();
	}
	
	public static long readLong(long g[], long bitIndex){
		throw new Todo();
	}
	
	public static Zing emptyMap = null; //FIXME
	
	public static long longInByteArrayByteAligned(byte a[], int byteIndex){
		long g = 0;
		for(int i=0; i<8; i++){
			g = (g<<8) & (a[byteIndex+i]&0xff);
		}
		return g;
	}
	
	/** Bytes are all 0 where index is outside array */
	public static long longInByteArrayByteAlignedCanHangOffEnds(byte a[], int byteIndex){
		long g = 0;
		for(int i=0; i<8; i++){
			int bi = byteIndex+i;
			byte b = 0 <= bi && bi < a.length ?  a[bi] : (byte)0;
			g = (g<<8) & (b&0xff);
		}
		return g;
	}
	
	public static long longInByteArrayBitAligned(byte a[], long bitIndex){
		throw new Todo();
	}
	
	/** Does for Zing what String.intern() does for String, except by hash.
	Uses ZingRoot synchronized WeakHashMap system.
	<br><br>
	FIXME? Is this paradox solvable? get(z.hash()) allows all the hash algorithms,
	while z.hash(defaultHashAlg) proves dedup.
	Choose how to dedup. Every call of hash(byte) will add that hash to WeakHashMap namespace
	so can look up some value by any of them, but if hash 2 of the same large bitstring
	by 2 hashAlg, they wont dedup. The forest will find them, but they wont dedup.
	*/
	public static Zing dedup(Zing z){
		if(z.isHash()) throw new Todo(
			"If you want to dedup hashes, add a second WeakHashMap<Zing,Zing> that maps a hash to itself");
		//return get(z.hash()); //SEE FIXME IN COMMENT OF THIS FUNC
		return get(z.hash(defaultHashAlg)); //SEE FIXME IN COMMENT OF THIS FUNC
		
		/*Zing h = z.hash(Hash.defaultHashAlg);
		Zing first = ZingRoot.get(h);
		if(first != null) return first;
		ZingRoot.put(z);
		return z;
		*/
	}
	
	/** Compares 2 longs as unsigned. Returns -1, 0, or 1, compatible with Comparator. */
	public static int compareUint64(long x, long y){
		boolean xNeg =  x < 0, yNeg = y < 0;
		if(xNeg && !yNeg) return 1; //x > y
		if(!xNeg && yNeg) return -1; //x < y
		if(x < y) return -1;
		if(x > y) return 1;
		return 0;
	}
	
	/** This can create invalid Zing and should only be used at child borders as header says *
	public Zing childCopyAt(long a[], long startAtBit, long endExclAtBit, boolean copy){
		"Use in LongArrayZing"
	}*/
	
	private static long randomPerJvm[];
	static{
		randomPerJvm = new long[100+Rand.strongRand.nextInt(21)]; //size 100-120
		for(int i=0; i<randomPerJvm.length; i++){
			randomPerJvm[i] = Rand.strongRand.nextLong();
		}
	}
	
	/** See comment in Zing interface about the hashCode() algorithm which must be used for all Zing
	TODO write here and in zing comment exactly what hash32 algorithm (is here and) must be used for all Zings.
	for consistency of Comparator and equals.
	This does not trigger lazy-eval-secureHash.
	Hash32 should be cached by childs constructor.
	Hash nonleafs by their childs since their bits may not exist yet.
	Leafs already know their bits so are hashed by those. 
	*/
	public static int hash32(Zing z){
		long g = 0;
		boolean notAtEnd = false;
		if(z.isMaplist()){
			long cyclesJ = z.sizeList();
			for(long i=0; i<cyclesJ; i++){
				//This doesnt trigger lazy-eval-secureHash, and hash32 should be cached by childs constructor
				g += z.listGet(i).hashCode() * randomPerJvm[(int)(i%randomPerJvm.length)];
			}
			if(z.isMap()) notAtEnd = true;
		}else{
			//This doesnt trigger lazy-eval-secureHash
			long cyclesJ = ((z.sizeBits()+63)/64);
			if(cyclesJ <= Integer.MAX_VALUE){ //optimized long aligned
				int cycles = (int)cyclesJ;
				for(int i=0; i<cycles; i++){
					g += z.ja(i) ^ randomPerJvm[i%randomPerJvm.length];
				}
			}else{ //slower bit aligned can handle bigger size
				for(long i=0; i<cyclesJ; i++){
					g += z.j(i<<6) ^ randomPerJvm[(int)(i%randomPerJvm.length)];
				}
			}
		}
		int h = ((int)g) ^ ((int)(g>>>32));
		return notAtEnd ? ~h : h;
	}
	
	static{ lg("TODO handle classes that already exist and have been auto renamed to a hashname"); }
	
	public static Predicate<String> javaFilter = (String code)->{
		return false; //TODO
	};
	
	public static Predicate<String> javascriptFilter = (String code)->{
		return false; //TODO
	};
	
	/** return some kind of java lambda */
	public static final Function<String,Object> codeCompileJava = (String code)->{
		throw new Todo();
	};
	
	/** return some kind of java lambda despite it being javascript code */
	public static final Function<String,Object> codeCompileJavascript = (String code)->{
		throw new Todo();
	};
	
	/** Param is string of code prefixed with "java:", "javascript:", etc.
	Returns true if allow the code to run.
	Only allows certain kinds of code,
	such as those proven to only do math and call certain zing funcs.
	The purpose is to sandbox code so it can be run without trust.
	Also enforce computeMoney/memoryMoney/econacyc/etc here.
	Or whatever are the rules for what untrusted code can run.
	It automaticly becomes trusted by proving it can only do certain things.
	The filter doesnt have to understand the code, only verify each small part.
	Make sure to filter all lambda code in case its added in new version of javassist.
	Other than that, my gigalinecompile java parser,
	which is designed only to detect lvalues and rvalues,
	is a good place to start building a partial parser for this. 
	*/
	public static final Predicate<String> codeFilter = (String code)->{
		if(code.startsWith("java:")){
			return javaFilter.test(code.substring("java:".length()));
		}else if(code.startsWith("javascript:")){
			return javascriptFilter.test(code.substring("javascript:".length()));
		}else{
			return false;
		}
	};
	
	/** Param is string of code prefixed with "java:", "javascript:", etc.
	Return is a java lambda such as Function or DoubleBinaryOperator.
	You should only call this if filter(code) returns true.
	TODO for java code, only accept code in java8 lambda syntax that defines no inner lambdas,
	and automatically translate that to java5to7 syntax and give that to javassist.
	First param is always named x, and second param (if it exists) is always named y.
	Always import zing package.
	Forexample, code that implements DoubleBinaryOperator as normal interface.
	*/
	public static final Function<String,Object> codeCompile = (String code)->{
		if(code.startsWith("java:")){
			return codeCompileJava.apply(code.substring("java:".length()));
		}else if(code.startsWith("javascript:")){
			return codeCompileJavascript.apply(code.substring("javascript:".length()));
		}else{
			throw new Err("Language unknown in code: "+code);
		}
	};
	
	/** throws if not pass filter or compile error */
	public static final Function<String,Object> codeFilterThenCompile = (String code)->{
		if(!codeFilter.test(code)) throw new Err("Permission denied for code: "+code);
		return codeCompile.apply(code);
	};
	
	public static Zing bitstring(long[] content){
		if(content.length < 1024){ //!isBig
			if(content.length == 0) throw new Todo("empty leaf Zing");
			long g[] = new long[content.length+1];
			long sizeBits = 32+(content.length<<6);
			//4 mask bits and 12 sizeList are all 0
			g[0] |= (sizeBits<<32); //16 sizeBits bits
			for(int i=0; i<content.length; i++){
				g[i] |= content[i]>>>32;
				g[i+1] |= content[i]<<32;
			}
			return new LongArrayZing(g);
		}else{ //isBig
			long sizeBits = 128+content.length<<64;
			long g[] = new long[2+content.length];
			//4 mask bits and 60 sizeList bits are all 0, so leave g[0] as 0
			g[1] = sizeBits;
			System.arraycopy(content, 0, g, 2, content.length);
			return new LongArrayZing(g);
		}
	}
	
	public static List<String> toListOfString(Zing z){
		List<String> list = new ArrayList();
		for(Zing child : z.iterList()){
			list.add(s(child));
		}
		return Collections.unmodifiableList(list);
	}
	
	public static Zing toZingFromListOfString(List<String> list){
		Zing childs[] = new Zing[list.size()];
		for(int i=0; i<childs.length; i++){
			childs[i] = s(list.get(i));
		}
		return new LongArrayZing(false, childs);
	}
	
	public static long longInByteArrayAtByteIndex(byte a[], int byteIndex){
		long g = 0;
		for(int i=0; i<8; i++){
			g = (g<<8) | (a[byteIndex+i]&0xff);
		}
		return g;
	}
	
	public static long longInByteArrayAtByteIndexOrZerosWhereOutOfRange(byte a[], int byteIndex){
		long g = 0;
		for(int i=0; i<8; i++){
			if(0 <= byteIndex+i && byteIndex+i < a.length) g |= (a[byteIndex+i]&0xff);
			if(i != 7) g <<= 8;
		}
		return g;
	}
	
	/** copies from offset to offset+7 */
	public static void copyLongIntoByteArrayAtByteOffset(long data, byte a[], int byteOffset){
		for(int i=7; i>=0; i--){
			a[byteOffset+i] = (byte)data;
			data >>>= 8;
		}
	}
	
	/** prefixes header and returns a Zing of that leaf bitstring. Not all leafs are multiple of 8 bits. */
	public static Zing leafFromContentBytes(byte content[]){
		//FIXME check this math
		long sizeBits = content.length < ((1<<13)-4-1) ? 32+content.length*8L : 128+content.length*8L;
		long g[] = new long[(int)((sizeBits+63)/64)]; //round up
		boolean isBig = 0xffff < sizeBits;
		//4 mask bits are all 0
		int byteOffset;
		if(isBig){
			byteOffset = 16;
			//uint60 sizeList is 0
			g[1] = sizeBits;
			for(int i=2; i<g.length-1; i++){
				g[i] = longInByteArrayAtByteIndex(content, (i<<3)-byteOffset); //faster
			}
		}else{ //!isBig
			byteOffset = 4;
			//uint12 sizeList is 0
			g[0] = (sizeBits<<32) | longInByteArrayAtByteIndexOrZerosWhereOutOfRange(content,byteOffset-8); //slower but only once
			for(int i=1; i<g.length-1; i++){
				g[i] = longInByteArrayAtByteIndex(content, (i<<3)-byteOffset); //faster
			}
		}
		if((isBig && 2 < g.length) || (!isBig && 1 < g.length)){
			g[g.length-1] = longInByteArrayAtByteIndexOrZerosWhereOutOfRange(content, ((g.length-1)<<3)-byteOffset); //slower but only once
		}
		return new LongArrayZing(g);
	}
	
	public static byte[] contentBytesFromLeaf(Zing leaf){
		if(!leaf.isLeaf()) throw new Err("Not a leaf: "+leaf);
		if((leaf.sizeBits()&7) != 0) throw new Err(
			"Not a multiple of 8 bits, which is allowed in general but not in this func. leaf: "+leaf);
		int byteOffset = leaf.isBig() ? 16 : 4;
		long contentSizeBytes = (leaf.sizeBits()>>>3)-byteOffset;
		if(Integer.MAX_VALUE < contentSizeBytes) throw new Err(
			"Leaf is too big to fit in max byte array,"
			+" which is ok in general but not for this func. sizeBits="+leaf.sizeBits());
		byte b[] = new byte[(int)contentSizeBytes];
		for(int i=0; i<b.length; i++){
			b[i] = leaf.ba(byteOffset+i);
		}
		return b;
	}
	
	/** If not a multiple of 8 bytes, the remaining bytes in last long are all 0 */
	public static long[] toLongArrayFromByteArray(byte b[]){
		long g[] = new long[(b.length+7)/8]; //round up
		for(int i=0; i<g.length-1; i++){
			g[i] = longInByteArrayAtByteIndex(b, i<<3); //faster
		}
		g[g.length-1] = longInByteArrayAtByteIndexOrZerosWhereOutOfRange(b, (g.length-1)<<8); //slower but only once
		return g;
	}
	
	/** offset is the first bit index in long[]. Does not modify bits outside the range offset to offset+z.sizeBits()-1 */
	public static void copyZingIntoLongArray(Zing z, long g[], long offset){
		//FIXME optimize by copying long at a time except at ends
		long end = z.sizeBits();
		for(long i=0; i<end; i++){
			copyBitIntoLongArray(z.z(i), g, offset+i);
		}
	}
	
	/** offset is the first bit index in long[]. Does not modify bits outside the range offset to offset+15 */
	public static void copyShortIntoLongArray(short data, long g[], long offset){
		//FIXME optimize by copying long at a time except at ends
		for(long i=0; i<16; i++){
			boolean zData = (data>>>(15-i)) != 0;
			copyBitIntoLongArray(zData, g, offset+i);
		}
	}
	
	/** offset is the first bit index in long[]. Does not modify bits outside the range offset to offset+63 */
	public static void copyLongIntoLongArray(long data, long g[], long offset){
		//FIXME optimize by copying long at a time except at ends
		for(long i=0; i<64; i++){
			boolean zData = (data>>>(63-i)) != 0;
			copyBitIntoLongArray(zData, g, offset+i);
		}
	}
	
	/** Slow but useful for getting things done at ends of large ranges and for prototyping before optimize. */
	public static void copyBitIntoLongArray(boolean data, long g[], long offset){
		int gIndex = (int)(offset>>>6);
		long mask = 1L<<(63-(offset&63));
		if(data) g[gIndex] |= mask;
		else g[gIndex] &= ~mask;
	}
	
	public static long[] longArrayFromHeaderAndCopyChilds(boolean isMap, Zing... childs){
		if(isMap && childs.length == 0) throw new Err(
			"The normed form of an empty map is an empty list because list is abstractly a map whose keys are all longs. Must norm.");
		//FIXME check for nonempty map whose keys are all longs (32+64 bits each) and counting as in a list, and throw for not being normed.
		for(int i=0; i<childs.length; i++){
			childs[i] = ptr(childs[i]); //idempotent
		}
		long sumChildsBitSize = 0;
		for(Zing child : childs){
			sumChildsBitSize += child.sizeBits();
		}
		long sizeBitsIfSmall = 32+childs.length*16+sumChildsBitSize;
		boolean isBig = 0xffff < sizeBitsIfSmall;
		long sizeBits = isBig ? 128+childs.length*64+sumChildsBitSize : sizeBitsIfSmall;
		long arraySize = (sizeBits+63)/64; //round up
		if(Integer.MAX_VALUE < arraySize) throw new Err("new long["+arraySize+" exceeds maxint]");
		long g[] = new long[(int)arraySize];
		g[0] = isMap ? ZingRoot.maskIsMap : ZingRoot.maskIsList;
		if(isBig){
			g[0] |= ZingRoot.maskIsBig;
			g[0] |= childs.length; //after 4 mask bits, uint60 sizeList
			g[1] = sizeBits; //after 64 bits, uint64 sizeBits. This is overwritten by equal value by cumSize if list is 
		}else{
			g[0] |= ((long)childs.length)<<48; //after 4 mask bits, uint12 sizeList
			g[0] |= sizeBits<<32; //after 16 bits, uint16 sizeBits
		}
		//Cumulative bit size (the first bit index after each child ends) is in reverse order of the childs
		//so parent bit size is at either of 2 constant locations depending on isBig.
		long cumSize = sizeBits;
		for(int i=0; i<childs.length; i++){
			Zing child = childs[childs.length-1-i];
			if(isBig) g[2+i] = cumSize;
			else copyShortIntoLongArray((short)cumSize, g, 32+16*i);
			//Each endExcl is start of another except the last in this loop which is instead header size
			cumSize -= child.sizeBits();
			copyZingIntoLongArray(child, g, cumSize);
		}
		return g;
	}
	
	private static Zing defaultFuncs;
	
	public static final Zing defaultFuncs(){
		if(defaultFuncs == null){
			Map<Zing,Zing> map = new HashMap();
			map.put(s("lg"), s("(String x)->{System.out.println(x);}"));
			defaultFuncs = new LongArrayZing(map);
		}
		return defaultFuncs;
	}

}