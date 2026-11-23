import threading,time,unittest
from unittest.mock import Mock,patch
from dataclasses import replace
from PIL import Image,ImageDraw
from context_operator.core.contracts import Rect,Element
from context_operator.providers.uia_reader import distinct_elements,ROLES
from context_operator.providers.visual_regions import VisualFrame,digest,regions,verified,unique_verified
from context_operator.providers.visual_cache import VisualCache
from context_operator.controller import Controller

class RegionTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(300,180),'black');ImageDraw.Draw(self.image).rectangle((20,20,60,60),fill='white')
        self.rect=Rect(-300,100,300,180);self.frame=VisualFrame((7,8,self.rect,1),self.rect,digest(self.image))
        self.result={'items':[{'rect':[20,20,40,40],'source':'icon','label':'Plus'}]}
        self.items=regions(self.frame,self.image,self.result)
    def test_unrelated_animation_does_not_erase_button_but_changed_button_or_margin_does(self):
        animated=self.image.copy();ImageDraw.Draw(animated).rectangle((120,40,190,150),fill='blue')
        self.assertEqual(verified(self.items,animated,self.rect),self.items)
        for point in ((40,40),(16,20)):
            changed=animated.copy();changed.putpixel(point,(255,0,0))
            self.assertEqual(verified(self.items,changed,self.rect),())
    def test_gaze_roi_can_move_without_losing_still_visible_unchanged_button(self):
        shifted=self.image.crop((10,10,200,140));rect=Rect(-290,110,190,130)
        self.assertEqual(verified(self.items,shifted,rect),self.items)
        clipped=self.image.crop((30,0,300,180))
        self.assertEqual(verified(self.items,clipped,Rect(-270,100,270,180)),())
    def test_outside_image_predictions_are_not_executable_regions(self):
        invalid={'items':[{'rect':[-10,20,40,40],'source':'icon','label':'bad'},
                          {'rect':[290,20,40,40],'source':'icon','label':'bad'}]}
        self.assertEqual(regions(self.frame,self.image,invalid),())
    def test_busy_inference_keeps_verified_region_and_owner_epoch_isolation(self):
        release=threading.Event();entered=threading.Event();stopped=threading.Event();calls=[];response=self.result
        class Client:
            def ready(self,stop):pass
            def observe(self,image,stop):
                calls.append(1)
                if len(calls)>1:entered.set();release.wait(2)
                return response
            def close(self):pass
        cache=VisualCache(stopped,Client)
        try:
            owner=self.frame.owner;end=time.monotonic()+1
            found=cache.lookup_frame(owner,self.rect,self.image)
            while not found and time.monotonic()<end:
                time.sleep(.01);found=cache.lookup_frame(owner,self.rect,self.image)
            self.assertEqual(found,self.items)
            animated=self.image.copy();animated.putpixel((200,100),(2,3,4))
            self.assertEqual(cache.lookup_frame(owner,self.rect,animated),self.items)
            self.assertTrue(entered.wait(.5))
            self.assertEqual(cache.lookup_frame(owner,self.rect,animated),self.items)
            self.assertEqual(cache.lookup_frame((7,9,self.rect,1),self.rect,animated),())
            self.assertEqual(cache.lookup_frame((7,8,self.rect,2),self.rect,animated),())
            with patch('context_operator.providers.visual_cache.time.monotonic',return_value=time.monotonic()+13):
                self.assertEqual(cache.lookup_frame(owner,self.rect,animated),())
                self.assertEqual(cache.lookup_frame(owner,self.rect,self.image),self.items)
        finally:release.set();cache.close()

    def test_returning_gaze_crops_do_not_accumulate_jittered_copies(self):
        cache=VisualCache(threading.Event(),Mock());self.addCleanup(cache.close)
        for n in range(24):
            frame=self.frame if n==23 else replace(self.frame,pixels=bytes([n])*12)
            output={'items':[{'rect':[18+n%5,18+n//5,40,40],'source':'icon','label':str(n)}]}
            cache.cache[frame]=(100.,regions(frame,self.image,output))
        self.assertEqual(len({p.rect for _,items in cache.cache.values() for p in items}),24)
        with patch('context_operator.providers.visual_cache.time.monotonic',return_value=100.5), \
             patch('context_operator.providers.visual_regions.digest',wraps=digest) as hash_patch:
            found=cache.lookup_frame(self.frame.owner,self.rect,self.image)
        self.assertEqual(len(found),1);self.assertEqual(found[0].label,'23')
        self.assertEqual(hash_patch.call_count,1)  # Older equivalent patches are not rehashed.

    def test_duplicate_filter_preserves_neighbours_nested_controls_and_verified_fallback(self):
        output={'items':[{'rect':[20,20,40,40],'source':'icon','label':'outer'},
                         {'rect':[32,32,10,10],'source':'icon','label':'inner'},
                         {'rect':[63,20,40,40],'source':'icon','label':'neighbour'}]}
        items=regions(self.frame,self.image,output)
        stale=replace(items[0],pixels=b'changed')
        self.assertEqual(unique_verified((stale,*items),self.image,self.rect),items)

    def test_small_icon_full_plus_minus_three_pixel_jitter_stays_one_target(self):
        for size in (16,24,32,40):
            candidates=[]
            for offset in [-3,-2,-1,0,1,2,3,2,1,0,-1,-2]*2:
                output={'items':[{'rect':[50+offset,50+offset,size,size],'source':'icon','label':'same'}]}
                candidates.extend(regions(self.frame,self.image,output))
            self.assertEqual(len(unique_verified(reversed(candidates),self.image,self.rect)),1,size)

class SemanticCoverageTests(unittest.TestCase):
    def test_repeated_accessibility_paths_do_not_paint_the_same_control_twice(self):
        first=Element('path-a','Back','Button',Rect(20,30,40,40),1)
        repeated=replace(first,id='path-b')
        neighbour=replace(first,id='path-c',rect=Rect(65,30,40,40))
        result=distinct_elements((first,repeated,neighbour))
        self.assertEqual(len(result),2);self.assertEqual({e.rect for e in result},{first.rect,neighbour.rect})
    def test_public_text_remains_but_captions_do_not_steal_button_or_editor(self):
        self.assertIn('Text',ROLES)
        button=Element('b','Plus','Button',Rect(10,10,100,40),1)
        caption=Element('caption','Plus','Text',Rect(20,20,60,20),1)
        editor=Element('edit','Compose','Edit',Rect(10,70,300,100),1)
        placeholder=Element('placeholder','Message','Text',Rect(20,80,160,20),1)
        text=Element('paragraph','Read this','Text',Rect(10,200,300,50),1)
        self.assertEqual(distinct_elements((button,caption,editor,placeholder,text)),(button,editor,text))
    def test_broad_list_does_not_swallow_small_visual_button(self):
        row=Element('row','Card','ListItem',Rect(0,0,800,200),1)
        icon=Element('icon','Icon','Icon',Rect(500,50,30,30),1,'icon')
        self.assertFalse(Controller._covers_visual(row,icon))
        button=replace(row,role='Button',rect=Rect(495,45,40,40))
        self.assertTrue(Controller._covers_visual(button,icon))
    def test_editor_does_not_swallow_icon_and_one_word_does_not_replace_whole_ocr_line(self):
        editor=Element('editor','Compose','Edit',Rect(0,0,600,200),1)
        icon=Element('send','Send','Icon',Rect(500,100,30,30),1,'icon')
        self.assertFalse(Controller._covers_visual(editor,icon))
        line=Element('line','One whole line','Text',Rect(0,30,100,20),1,'ocr')
        word=Element('word','whole','Text',Rect(40,30,20,20),1)
        self.assertFalse(Controller._covers_visual(word,line))
        self.assertTrue(Controller._covers_visual(editor,line))

if __name__=='__main__':unittest.main()
